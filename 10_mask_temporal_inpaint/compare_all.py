"""
Comparaison globale — Toutes les approches sur 3 frames témoins

Approches comparées :
  Original  — image brute
  A         — Homographie + médiane (N=5)
  D         — Lowpass temporel gaussien
  G         — RAFT + masque inclusif + N=15
  H         — Décomposition base/détail (sigma=25)
  I         — Pyramide Laplacienne (L=4, fine=1, N=9)

Sorties :
  comparaison_all_approaches.jpg  — grille frames × approches
  sharpness_report.txt            — tableau quantitatif
"""
import cv2
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

# ── Sources ────────────────────────────────────────────────────────────────

VIDEOS = {
    'Original':    os.path.join(ROOT, '01_extraction', 'subclip_5s.mp4'),
    'A_Homogr':    os.path.join(ROOT, '06_homography_median', 'result_homography_N5.mp4'),
    'D_Lowpass':   os.path.join(ROOT, '09_temporal_lowpass', 'result_lowpass.mp4'),
    'G_RAFT':      os.path.join(BASE, 'result_G_final_N15.mp4'),
    'H_BaseDetail':os.path.join(BASE, 'result_H_decompose.mp4'),
    'I_Pyramid':   os.path.join(BASE, 'result_I_pyramid_N9.mp4'),
    'J_Selective': os.path.join(BASE, 'result_J_selective_N9.mp4'),
}

# NB : les approches B et C sont dans 07/08 — pas de MP4 sauvegardé car résultats
# inférieurs → on les omet ici. À réactiver si besoin.

SCALE = 0.25            # les vidéos résultat sont déjà à 0.25 ; l'original non
TEST_FRAMES = [15, 75, 130]
LABEL_HEIGHT = 38       # hauteur de la bannière de label


def read_frame(path, frame_idx, scale=None):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if scale is not None:
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    return frame


def add_label(img, text, color=(255, 255, 255)):
    """Ajoute une barre de label en haut de l'image."""
    h, w = img.shape[:2]
    bar = np.zeros((LABEL_HEIGHT, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, LABEL_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    approaches = list(VIDEOS.keys())
    n_app = len(approaches)
    n_frames = len(TEST_FRAMES)

    # ── Chargement ────────────────────────────────────────────────────────
    # images[approche][frame_idx] = img
    images = {a: {} for a in approaches}
    for name, path in VIDEOS.items():
        if not os.path.exists(path):
            print(f"[MISSING] {name}: {path}")
            continue
        scale = SCALE if name == 'Original' else None
        for fi in TEST_FRAMES:
            img = read_frame(path, fi, scale)
            if img is not None:
                images[name][fi] = img
            else:
                print(f"[WARN] Impossible de lire frame {fi} dans {path}")

    # ── Taille de référence ───────────────────────────────────────────────
    ref_img = next(iter(images['Original'].values()))
    h, w = ref_img.shape[:2]

    # ── Rapport de netteté ───────────────────────────────────────────────
    print(f"\n{'':>14}" + "".join(f"{a:>14}" for a in approaches))
    print("-" * (14 + 14 * n_app))

    report_lines = ["Sharpness (Laplacian variance) — higher is better\n"]
    report_lines.append(f"{'Frame':>8}" + "".join(f"{a:>16}" for a in approaches) + "\n")
    report_lines.append("-" * (8 + 16 * n_app) + "\n")

    orig_sharps = {}
    for fi in TEST_FRAMES:
        row_str = f"{fi:>8}"
        orig_sharp = laplacian_variance(images['Original'][fi]) if fi in images.get('Original', {}) else 0
        orig_sharps[fi] = orig_sharp
        for name in approaches:
            img = images[name].get(fi)
            if img is None:
                row_str += f"{'N/A':>16}"
            else:
                sharp = laplacian_variance(img)
                if name == 'Original':
                    row_str += f"{sharp:>16.1f}"
                else:
                    delta = (sharp - orig_sharp) / orig_sharp * 100 if orig_sharp > 0 else 0
                    row_str += f"{sharp:>10.1f}({delta:+.1f}%)"
        print(row_str)
        report_lines.append(row_str + "\n")

    report_path = os.path.join(BASE, 'sharpness_report.txt')
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f"\nRapport sauvegardé : {report_path}")

    # ── Grille visuelle ───────────────────────────────────────────────────
    # Lignes = approches, colonnes = frames témoins
    # En-tête colonne = numéro de frame
    cell_w, cell_h = w, h  # toutes les frames sont à la même résolution

    rows_imgs = []

    for name in approaches:
        row_cells = []
        for fi in TEST_FRAMES:
            img = images[name].get(fi)
            if img is None:
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(cell, 'N/A', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            else:
                if img.shape[:2] != (cell_h, cell_w):
                    img = cv2.resize(img, (cell_w, cell_h))
                cell = img.copy()

            label = f"{name} — frame {fi}"
            if name != 'Original' and orig_sharps.get(fi, 0) > 0:
                sharp = laplacian_variance(cell)
                delta = (sharp - orig_sharps[fi]) / orig_sharps[fi] * 100
                label += f"  sharp:{delta:+.1f}%"
            cell = add_label(cell, label, color=(0, 255, 0) if name != 'Original' else (100, 200, 255))
            row_cells.append(cell)

        rows_imgs.append(np.hstack(row_cells))

    grid = np.vstack(rows_imgs)

    out_path = os.path.join(BASE, 'comparaison_all_approaches.jpg')
    cv2.imwrite(out_path, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"Grille de comparaison : {out_path}  ({grid.shape[1]}×{grid.shape[0]})")


if __name__ == '__main__':
    main()
