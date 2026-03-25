import cv2
import numpy as np

def single_scale_retinex(img_float, sigma):
    """Retinex à une seule échelle: log(I) - log(I * G(sigma))"""
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    blurred = np.maximum(blurred, 1e-6)
    retinex = np.log10(img_float + 1e-6) - np.log10(blurred)
    return retinex

def multi_scale_retinex(img_channel, sigmas=[15, 80, 250]):
    """Moyenne des Retinex sur plusieurs échelles."""
    img_float = img_channel.astype(np.float32)
    result = np.zeros_like(img_float)
    for sigma in sigmas:
        result += single_scale_retinex(img_float, sigma)
    result /= len(sigmas)
    return result

def normalize_retinex(msr):
    """Normalise le résultat Retinex en image 0-255."""
    # Normalisation perceptuelle avec percentiles pour éviter les outliers
    p2 = np.percentile(msr, 2)
    p98 = np.percentile(msr, 98)
    msr_norm = np.clip((msr - p2) / (p98 - p2 + 1e-6), 0, 1)
    return (msr_norm * 255).astype(np.uint8)

def retinex_on_image(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    # Appliquer MSR sur chaque canal B,G,R indépendamment
    channels_out = []
    for i in range(3):
        channel = img[:, :, i]
        msr = multi_scale_retinex(channel.astype(np.float64))
        channel_corrected = normalize_retinex(msr)
        channels_out.append(channel_corrected)
    
    result = cv2.merge(channels_out)
    
    # Comparaison côte à côte, réduite à 50%
    img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    res_small = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)
    
    cv2.putText(img_small, "Avant", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    cv2.putText(res_small, "Apres MSR Retinex", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    
    final = np.hstack((img_small, res_small))
    cv2.imwrite(out_path, final)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    retinex_on_image("comparaison_1_originale.jpg", "test_retinex_msr.jpg")
