"""
Approche I — Décomposition Laplacien Pyramide multi-niveaux + médiane temporelle sélective

Problème de H (sigma=25) :
  Les caustiques fines et brillantes (ex: fond sableux) ont des composantes
  haute-fréquence qui tombent dans la couche "détail" → pas supprimées.

Solution : pyramide de Laplace à N niveaux.
  - Niveau 0 (plus fin) : texture pure à haute fréquence → CONSERVÉ intact
  - Niveaux 1..L-1   : bandes intermédiaires (contient les caustiques à toutes
                        échelles) → médiane temporelle appliquée
  - Niveau L (fond)   : illumination basse fréquence → médiane temporelle

De cette façon, seule la vraie texture haute fréquence est préservée, et
TOUTES les échelles de caustiques sont supprimées temporellement.

Paramètres :
  PYRAMID_LEVELS   : nombre de niveaux (4 = bon compromis)
  KEEP_FINE_LEVELS : nombre de niveaux fins conservés (1 = seulement le plus fin)
  N                : taille de la fenêtre temporelle
"""
import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE = 0.25
N = 9                # window size (nb frames voisines)
PYRAMID_LEVELS = 4   # niveaux de la pyramide (plus = plus de séparation)
KEEP_FINE_LEVELS = 1 # nb de niveaux fins à garder intact (1 = seulement le + fin)
TEST_INDICES = [15, 75, 130]


def load_frames(video_path, scale):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (0, 0), fx=scale, fy=scale))
    cap.release()
    return frames, fps


def build_laplacian_pyramid(img_f, levels):
    """
    Construit une pyramide de Laplace.
    Retourne une liste [lap_0, lap_1, ..., lap_{L-1}, gauss_L].
      lap_k = gauss_k - upsample(gauss_{k+1})
      gauss_L = image la plus floue (fond)
    """
    gauss = [img_f.copy()]
    for _ in range(levels):
        g = cv2.pyrDown(gauss[-1])
        gauss.append(g)

    pyramid = []
    for k in range(levels):
        h, w = gauss[k].shape[:2]
        up = cv2.pyrUp(gauss[k + 1], dstsize=(w, h))
        lap = gauss[k] - up
        pyramid.append(lap)
    pyramid.append(gauss[-1])  # dernier niveau : gausse basse fréquence
    return pyramid  # longueur = levels + 1


def reconstruct_from_pyramid(pyramid):
    """Reconstruit l'image depuis la pyramide."""
    levels = len(pyramid) - 1
    result = pyramid[-1].copy()
    for k in range(levels - 1, -1, -1):
        h, w = pyramid[k].shape[:2]
        up = cv2.pyrUp(result, dstsize=(w, h))
        result = up + pyramid[k]
    return result


def align_homography(ref, neighbor):
    """Aligne neighbor sur ref avec SIFT + homographie RANSAC (sur BGR uint8)."""
    sift = cv2.SIFT_create(nfeatures=2000)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    neigh_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(neigh_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return neighbor
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return neighbor
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return neighbor
    h, w = ref.shape[:2]
    return cv2.warpPerspective(neighbor, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


def process_frame_pyramid(frames, center_idx, half,
                          levels=PYRAMID_LEVELS,
                          keep_fine=KEEP_FINE_LEVELS):
    """
    Traite une frame avec la décomposition pyramidale.
    keep_fine : nombre de niveaux fins (indices 0..keep_fine-1) conservés sans médiane.
    Les niveaux [keep_fine .. levels] subissent la médiane temporelle.
    """
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)
    ref = frames[center_idx]

    # Aligner les voisins (sur l'image originale uint8)
    aligned = [ref]
    for i in range(start, end):
        if i == center_idx:
            continue
        a = align_homography(ref, frames[i])
        aligned.append(a)

    # Convertir en float32
    aligned_f = [f.astype(np.float32) for f in aligned]

    # Construire les pyramides pour tous les voisins alignés
    pyramids = [build_laplacian_pyramid(f, levels) for f in aligned_f]

    # pyramids[frame][level] → tableau (h, w, 3)
    # Pour chaque niveau, créer une pile et prendre la médiane
    # SAUF pour les niveaux fins (0..keep_fine-1)
    ref_pyramid = pyramids[0]
    result_pyramid = []

    for lvl in range(levels + 1):  # levels niveaux Laplace + 1 fond gaussien
        if lvl < keep_fine:
            # Conserver le niveau fin de la frame de référence intact
            result_pyramid.append(ref_pyramid[lvl])
        else:
            # Médiane temporelle sur ce niveau
            stack = np.stack([p[lvl] for p in pyramids], axis=0)
            med = np.median(stack, axis=0)
            result_pyramid.append(med)

    # Reconstruire
    result_f = reconstruct_from_pyramid(result_pyramid)
    return np.clip(result_f, 0, 255).astype(np.uint8)


def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    print(f"=== Approche I : Pyramide Laplacienne (L={PYRAMID_LEVELS}, keep_fine={KEEP_FINE_LEVELS}, N={N}) ===")
    t0 = time.time()
    frames, fps = load_frames(VIDEO_PATH, SCALE)
    print(f"{len(frames)} frames chargées en {time.time()-t0:.1f}s")

    half = N // 2

    # ── Test sur 3 frames ────────────────────────────────────────────────────
    rows = []
    print(f"\n{'Frame':>8} {'Sharp_orig':>12} {'Sharp_result':>14} {'Delta%':>10}")
    for idx in TEST_INDICES:
        t1 = time.time()
        result = process_frame_pyramid(frames, idx, half)
        dt = time.time() - t1

        sharp_orig  = laplacian_variance(frames[idx])
        sharp_result = laplacian_variance(result)
        delta = (sharp_result - sharp_orig) / sharp_orig * 100
        print(f"{idx:>8} {sharp_orig:>12.1f} {sharp_result:>14.1f} {delta:>+9.1f}%  ({dt:.1f}s)")

        orig = frames[idx].copy()
        cv2.putText(orig,   f'Avant {idx}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, f'Apres I (L={PYRAMID_LEVELS},fine={KEEP_FINE_LEVELS})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rows.append(np.hstack([orig, result]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_I_pyramid.jpg')
    cv2.imwrite(out_path, comparison, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"\nSauvegardé : {out_path}")

    # ── Vidéo complète ───────────────────────────────────────────────────────
    print(f"\nTraitement vidéo complète ({len(frames)} frames)…")
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_I_pyramid_N9.mp4')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    for idx in range(len(frames)):
        result = process_frame_pyramid(frames, idx, half)
        out.write(result)
        if idx % 15 == 0:
            print(f"  Frame {idx}/{len(frames)}")

    out.release()
    print(f"Vidéo sauvegardée : {vid_path}  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
