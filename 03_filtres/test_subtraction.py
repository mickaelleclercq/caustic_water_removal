import cv2
import numpy as np

def fix_caustics_subtract(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    # On travaille en espace HSV pour les calculs de luminosité
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # 1. Obtenir le "fond" d'illumination (sans les caustiques)
    kernel_size = 21
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # On ferme pour homogénéiser
    # Mais les caustiques sont des *pics* clairs. Donc on utilise une Ouverture (Erosion puis dilatation)
    # L'ouverture retire les éléments clairs plus petits que le kernel.
    v_opened = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    
    # 2. Isoler les caustiques (Top-Hat)
    # Le tophat est exactement V - V_opened
    caustics = v - v_opened
    
    # 3. Atténuer subtilement 
    # Au lieu de faire un masque binaire brutal, on *soustrait* l'excédent de lumière
    # des zones affectées.
    # Pour ne pas trop assombrir, on applique un facteur.
    attenuation_factor = 0.8
    v_corrected = v - (caustics * attenuation_factor)
    
    # Optionnel: on augmente légèrement la saturation pour redonner vie à l'eau
    s_corrected = np.clip(s * 1.1, 0, 255)
    
    v_corrected = np.clip(v_corrected, 0, 255)
    hsv_corrected = cv2.merge((h, s_corrected, v_corrected))
    
    res = cv2.cvtColor(hsv_corrected.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Assemble côte à côte
    img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    res_resized = cv2.resize(res, (0, 0), fx=0.5, fy=0.5)
    
    # ajouter texte
    cv2.putText(img_resized, "Avant", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    cv2.putText(res_resized, "Apres Subtraction Subtile", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    
    final = np.hstack((img_resized, res_resized))
    cv2.imwrite(out_path, final)

if __name__ == "__main__":
    fix_caustics_subtract("comparaison_1_originale.jpg", "comparaison_subtraction.jpg")
