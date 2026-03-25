"""
Traitement vidéo complet : fenêtre glissante courte (N=9 frames) + ECC + médiane.

Approche : pour chaque frame de sortie, aligner les N-1 voisins directement sur
la frame centrale via ECC (EUCLIDEAN), puis prendre la médiane.
Les caustiques changent entre frames → disparaissent dans la médiane.
Le fond (coraux/sable) reste fixe après alignement → reste net.

Résolution sortie : 0.25× (960x540 depuis 4K)  — correspond à l'approche validée.
ECC calculé à la même résolution de sortie (pas de multi-scale, plus fiable).

Usage :
    python process_video_shortwindow.py [input] [output] [--scale 0.25] [--N 9] [--start 0] [--end -1]
"""
import cv2
import numpy as np
import time
import argparse


def ecc_align(ref_gray_small, frame_small, frame_full, ecc_to_full, criteria):
    """ECC calculé à basse résolution (rapide), warp appliqué à pleine résolution (précis).
    - ref_gray_small : référence en gris à basse résolution
    - frame_small    : voisin à basse résolution (pour l'ECC)
    - frame_full     : même voisin à résolution de sortie (pour le warp final)
    - ecc_to_full    : ratio résolution_sortie / résolution_ecc (ex: 2 si 480→960)
    """
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(ref_gray_small, gray, warp,
                                       cv2.MOTION_EUCLIDEAN, criteria)
        # Rescaler la translation pour l'appliquer à la résolution de sortie
        warp[0, 2] *= ecc_to_full
        warp[1, 2] *= ecc_to_full
        return cv2.warpAffine(frame_full, warp,
                              (frame_full.shape[1], frame_full.shape[0]),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)
    except cv2.error:
        return frame_full  # fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",   nargs="?", default="subclip_5s.mp4")
    parser.add_argument("output",  nargs="?", default="subclip_5s_shortwindow_N9.mp4")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Scale résolution sortie (0.25=960p, 0.5=1080p depuis 4K)")
    parser.add_argument("--N",     type=int, default=9,
                        help="Taille fenêtre glissante (impair recommandé)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=-1, help="-1 = toute la vidéo")
    args = parser.parse_args()

    N    = args.N
    half = N // 2

    # ECC : critères optimisés (30 itérations suffisent pour les frames proches)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 5e-3)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir {args.input}"); return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame   = total if args.end == -1 else min(args.end, total)
    start_frame = args.start

    w = int(orig_w * args.scale)
    h = int(orig_h * args.scale)

    # ECC à résolution divisée par 2 (rapide), warp appliqué à résolution de sortie (précis)
    # 1px d'erreur ECC → 2px à la sortie = acceptable
    ew = w // 2
    eh = h // 2
    ecc_to_full = w / ew   # = 2.0

    print(f"=== Short-Window ECC+Median (N={N}) ===")
    print(f"Input:  {args.input}  ({orig_w}x{orig_h} @ {fps:.1f}fps, {total} frames)")
    print(f"Output: {args.output}  ({w}x{h}, scale={args.scale})")
    print(f"Frames: {start_frame} → {end_frame}  ({end_frame-start_frame} frames)")
    print(f"Fenêtre ±{half} frames = ±{half/fps*1000:.0f} ms | ECC à {ew}x{eh} → warp à {w}x{h}")

    # ------------------------------------------------------------------ #
    # 1. Charger les frames nécessaires                                    #
    # ------------------------------------------------------------------ #
    load_start = max(0, start_frame - half)
    load_end   = min(total, end_frame + half)
    print(f"Chargement frames {load_start}→{load_end}...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, load_start)
    frames       = []   # résolution de sortie (w x h)
    frames_small = []   # résolution ECC (ew x eh)
    for _ in range(load_end - load_start):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (w, h))
        frames.append(frame)
        frames_small.append(cv2.resize(frame, (ew, eh)))
    cap.release()
    print(f"Chargé {len(frames)} frames.")

    # ------------------------------------------------------------------ #
    # 2. Traitement frame à frame avec ECC direct                         #
    # ------------------------------------------------------------------ #
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    t0 = time.time()
    n_out = 0

    for i in range(start_frame, end_frame):
        li = i - load_start
        ws = max(0, li - half)
        we = min(len(frames), li + half + 1)

        ref       = frames[li]
        ref_small = frames_small[li]
        ref_g     = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        aligned = []
        for j in range(ws, we):
            if j == li:
                aligned.append(ref)
            else:
                aligned.append(ecc_align(ref_g, frames_small[j], frames[j],
                                         ecc_to_full, criteria))

        stack  = np.stack(aligned, axis=0)
        median = np.sort(stack, axis=0)[len(aligned) // 2]

        out.write(median)
        n_out += 1

        if n_out % 30 == 0 or n_out == 1:
            elapsed  = time.time() - t0
            fps_proc = n_out / elapsed
            remain   = (end_frame - start_frame - n_out) / max(fps_proc, 0.001)
            print(f"  Frame {i}/{end_frame}  |  {fps_proc:.2f} fps"
                  f"  |  Restant: {remain/60:.1f} min"
                  f"  |  Win: {we-ws} frames")

    out.release()
    total_t = time.time() - t0
    print(f"\nTerminé! {n_out} frames en {total_t:.1f}s ({n_out/total_t:.2f} fps moyen)")
    print(f"Vidéo: {args.output}")


if __name__ == "__main__":
    main()
