import cv2
import numpy as np

def test_filters(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    # Redimensionner pour le test (plus rapide et plus facile à afficher côté par côté)
    scale = 0.4
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    
    # 1. Base (Original)
    res1 = img.copy()
    cv2.putText(res1, "1. Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 2. CLAHE sur luminosité (LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    res2 = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    cv2.putText(res2, "2. CLAHE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 3. Bilateral Filter fort (enlève le bruit HF en gardant les bords)
    res3 = cv2.bilateralFilter(img, d=15, sigmaColor=80, sigmaSpace=80)
    cv2.putText(res3, "3. Bilateral Filter", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # 4. Illumination Substraction (Fond estimé par grande ouverture)
    # Convertir en float
    img_float = img.astype(np.float32) / 255.0
    hsv_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
    v = hsv_float[:, :, 2]
    # Estimer le fond en ouvrant (enlève les reflets brillants fins)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background_v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    # Remplacer la luminance
    hsv_float[:, :, 2] = background_v
    res4 = cv2.cvtColor(hsv_float, cv2.COLOR_HSV2BGR)
    res4 = (res4 * 255).astype(np.uint8)
    # Recouvrir un peu de la couleur / contraste perdu
    res4 = cv2.addWeighted(res4, 1.2, np.zeros_like(res4), 0, 0)
    cv2.putText(res4, "4. Illumination Sub.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Assembler l'image en 2x2
    top = np.hstack((res1, res2))
    bottom = np.hstack((res3, res4))
    grid = np.vstack((top, bottom))
    
    cv2.imwrite("comparaison_grid.jpg", grid)

if __name__ == "__main__":
    test_filters("comparaison_1_originale.jpg")
