"""
Approche J — Pyramide Laplacienne + traitement sélectif du niveau fin

Problème résiduel de l'approche I (keep_fine=1) :
  Les arêtes fines et brillantes des caustiques tombent dans le niveau 0
  (bande la plus fine) qui est conservé intact → elles restent visibles.

Solution :
  Pour le niveau 0 (le plus fin), ne pas appliquer la médiane entière, mais :
    1. Calculer fine_orig = niveau 0 de la frame centrale
       Calculer fine_med  = médiane du niveau 0 sur les frames alignées
    2. Détecter les pixels "peak brillant" dans fine_orig :
         - valeur très positive → contribution caustique (arête brillante)
         - valeur normale (positive ET négative, équilibrée) → texture
    3. Masque progressif sur ces pics positifs
    4. fine_blend = fine_orig * (1-mask) + fine_med * mask
  Tous les autres niveaux reçoivent la médiane complète (comme dans I).

Résultat attendu :
  - Caustiques fines → supprimées (leurs arêtes positives → remplacées par médiane)
  - Texture corail / sable  → préservée (oscillations ± équilibrées, pas de masque)
  - Netteté : proche de I (-1 à -5 %)
"""
import cv2
import numpy as np
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '01_extraction', 'subclip_5s.mp4')
OUTPUT_DIR = os.path.dirname(__file__)
SCALE    = 0.25
N        = 9             # fenêtre temporelle
L        = 4             # niveaux de la pyramide
FINE_SIGMA = 2.5         # non utilisé — conservé pour rétro-compat
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
    gauss = [img_f.copy()]
    for _ in range(levels):
        gauss.append(cv2.pyrDown(gauss[-1]))
    pyramid = []
    for k in range(levels):
        h, w = gauss[k].shape[:2]
        up = cv2.pyrUp(gauss[k + 1], dstsize=(w, h))
        pyramid.append(gauss[k] - up)
    pyramid.append(gauss[-1])
    return pyramid


def reconstruct_from_pyramid(pyramid):
    result = pyramid[-1].copy()
    for k in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[k].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + pyramid[k]
    return result


def align_homography(ref, neighbor):
    sift = cv2.SIFT_create(nfeatures=2000)
    ref_gray  = cv2.cvtColor(ref,      cv2.COLOR_BGR2GRAY)
    neigh_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(ref_gray,   None)
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
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def caustic_mask_from_image(img_bgr, tophat_sizes=(11, 21, 35), threshold_sigma=1.5):
    """
    Détecte les zones de caustiques dans l'image originale via top-hat multi-échelle
    sur le canal V (HSV). Retourne un masque flou (H,W,1) float32 [0..1].

    Plus fiable que les statistiques du niveau fin car :
      - Opère sur l'image réelle (valeurs 0-255 absolues)
      - Top-hat : isole les pics lumineux locaux (caustiques) du fond
      - threshold_sigma plus bas = masque plus large (1.0 → ~10-15% de couverture)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    combined = np.zeros_like(v)
    for ks in tophat_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)
        combined = np.maximum(combined, tophat)

    mean_v = combined.mean()
    std_v  = combined.std()
    thresh = mean_v + threshold_sigma * std_v

    # Masque progressif (0=intact, 1=remplacer)
    mask = np.clip((combined - thresh) / (std_v + 1e-6), 0, 1)

    # Dilater légèrement pour couvrir les bords de la caustique
    mask_u8 = (mask * 255).astype(np.uint8)
    dil_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.dilate(mask_u8, dil_k, iterations=1)
    mask    = cv2.GaussianBlur(mask_u8.astype(np.float32), (9, 9), 2.5) / 255.0

    return mask[:, :, None]  # (H, W, 1)


def process_frame_J(frames, center_idx, half, levels=L, fine_sigma=FINE_SIGMA):
    start = max(0, center_idx - half)
    end   = min(len(frames), center_idx + half + 1)
    ref   = frames[center_idx]

    # Aligner les voisins
    aligned = [ref]
    for i in range(start, end):
        if i == center_idx:
            continue
        aligned.append(align_homography(ref, frames[i]))

    aligned_f = [f.astype(np.float32) for f in aligned]
    pyramids  = [build_laplacian_pyramid(f, levels) for f in aligned_f]
    ref_pyr   = pyramids[0]

    result_pyramid = []
    for lvl in range(levels + 1):
        stack = np.stack([p[lvl] for p in pyramids], axis=0)
        med   = np.median(stack, axis=0)

        if lvl == 0:
            # Niveau le plus fin : suppression sélective de l'EXCÈS positif seulement.
            #
            # Idée : si fine_orig >> fine_med à un pixel, c'est qu'une caustique
            # a ajouté un pic brillant dans ce niveau. On supprime uniquement cet excès,
            # sans toucher au reste (texture +/-, ombres, etc.).
            #
            #   excess   = max(fine_orig - fine_med, 0)   → pic positif causé par caustique
            #   progress = clip((excess - thresh) / thresh, 0, 1)  → masque progressif
            #   result   = fine_orig - progress * excess   → seulement les pics retirés
            fine_orig = ref_pyr[0].astype(np.float32)
            fine_med  = med.astype(np.float32)

            # Excès positif pixel-par-pixel
            excess = np.maximum(fine_orig - fine_med, 0)  # (H,W,3)

            # Seuil au 98e percentile de l'excès positif
            # → ne cible QUE les pics appartenant aux caustiques (~top 2% de pex = ~0.9% de tous)
            # → texture alignée : excess ≈ 0 → non touché
            # → caustiques : excess >> 0, bien au-delà du seuil
            exc_flat = excess.ravel()
            pos_only = exc_flat[exc_flat > 0]
            if len(pos_only) > 100:
                thresh = float(np.percentile(pos_only, 98))
            else:
                thresh = 50.0  # fallback si peu de pixels positifs

            progress = np.clip((excess - thresh) / (thresh + 1e-6), 0, 1)
            fine_result = fine_orig - progress * excess
            result_pyramid.append(fine_result)
        else:
            result_pyramid.append(med)

    return np.clip(reconstruct_from_pyramid(result_pyramid), 0, 255).astype(np.uint8)


def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    print(f"=== Approche J : Pyramide sélective (L={L}, fine_sigma={FINE_SIGMA}, N={N}) ===")
    t0 = time.time()
    frames, fps = load_frames(VIDEO_PATH, SCALE)
    print(f"{len(frames)} frames en {time.time()-t0:.1f}s")

    half = N // 2

    # ── Comparaison J vs I sur 3 frames ─────────────────────────────────────
    # Réimporter process_frame_pyramid de I pour comparer
    from process_I_pyramid import process_frame_pyramid as process_I

    rows = []
    print(f"\n{'Frame':>6}  {'Original':>10}  {'I (-1→-4%)':>12}  {'J_excès':>10}")
    print("-" * 50)

    for idx in TEST_INDICES:
        t1 = time.time()

        res_I = process_I(frames, idx, half)
        res_J = process_frame_J(frames, idx, half)
        dt = time.time() - t1

        so = laplacian_variance(frames[idx])
        si = laplacian_variance(res_I)
        sj = laplacian_variance(res_J)
        di = (si - so) / so * 100
        dj = (sj - so) / so * 100
        print(f"{idx:>6}  {so:>10.0f}  {si:>10.0f}({di:+.1f}%)  {sj:>10.0f}({dj:+.1f}%)  [{dt:.1f}s]")

        orig = frames[idx].copy()
        cv2.putText(orig,  f'Avant {idx}',          (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(res_I, f'I (keep_fine=1)',       (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)
        cv2.putText(res_J, f'J (excess suppression)', (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        rows.append(np.hstack([orig, res_I, res_J]))

    comparison = np.vstack(rows)
    out_path = os.path.join(OUTPUT_DIR, 'comparaison_J_selective.jpg')
    cv2.imwrite(out_path, comparison, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"\nSauvegardé : {out_path}")

    # ── Vidéo complète ───────────────────────────────────────────────────────
    print(f"\nTraitement vidéo complète ({len(frames)} frames)…")
    t0 = time.time()
    h, w = frames[0].shape[:2]
    vid_path = os.path.join(OUTPUT_DIR, 'result_J_selective_N9.mp4')
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for idx in range(len(frames)):
        out.write(process_frame_J(frames, idx, half))
        if idx % 15 == 0:
            print(f"  Frame {idx}/{len(frames)}")
    out.release()
    print(f"Vidéo : {vid_path}  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
