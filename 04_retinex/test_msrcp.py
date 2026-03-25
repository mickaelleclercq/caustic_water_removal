import cv2
import numpy as np

def msrcp(img, sigmas=(15, 80, 250), low_clip=0.01, high_clip=0.99):
    """
    Multi-Scale Retinex with Colour Preservation
    - Retinex est appliqué sur la luminance totale (pas canal par canal)
    - Les couleurs sont préservées en multipliant chaque canal par le même ratio
    """
    img_float = img.astype(np.float64)
    # Luminance = somme des canaux
    lumiance = np.sum(img_float, axis=2) / 3.0
    lumiance = np.maximum(lumiance, 1.0)
    
    # MSR sur la luminance
    luminance_retinex = np.zeros_like(lumiance)
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(lumiance, (0, 0), sigma)
        blurred = np.maximum(blurred, 1e-6)
        luminance_retinex += np.log(lumiance + 1e-6) - np.log(blurred)
    luminance_retinex /= len(sigmas)

    # Normaliser la luminance retinex
    p_low = np.percentile(luminance_retinex, low_clip * 100)
    p_high = np.percentile(luminance_retinex, high_clip * 100)
    luminance_retinex_norm = np.clip((luminance_retinex - p_low) / (p_high - p_low + 1e-6), 0, 1)

    # Appliquer le facteur de luminance corrigé à chaque canal (ratio pour préserver la couleur)
    # Pour chaque pixel, on multiplie par (luminance_target / luminance_original)
    scale = (luminance_retinex_norm * 255.0) / lumiance
    
    result = np.zeros_like(img_float)
    for i in range(3):
        result[:, :, i] = np.clip(img_float[:, :, i] * scale, 0, 255)
    
    return result.astype(np.uint8)

def test_msrcp(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: {img_path} not found")
        return
    
    print("Running MSRCP...")
    result = msrcp(img)
    
    img_small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    res_small = cv2.resize(result, (0,0), fx=0.5, fy=0.5)
    
    cv2.putText(img_small, "Avant", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    cv2.putText(res_small, "Apres MSRCP", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    
    final = np.hstack((img_small, res_small))
    cv2.imwrite(out_path, final)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    # Test sur les 3 frames
    for t in [5, 15, 30]:
        test_msrcp(f"screenshot_{t}s.jpg", f"result_msrcp_{t}s.jpg")
