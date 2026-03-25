import cv2
import numpy as np
import sys

def remove_caustics_single_frame(image_path, output_path):
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur : Impossible de lire {image_path}")
        return

    # Convertir en espace HSV pour isoler la luminosité (canal V)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 1. Isolation des formes lumineuses (Filtre Top-Hat)
    # Le noyau (kernel) détermine la taille maximale des caustiques à détecter
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)

    # 2. Seuillage pour créer le masque
    # Ne garder que les reflets SAILLANTS (ajuster le seuil de 30 à 100 selon l'intensité)
    _, mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)

    # Optionnel: nettoyer le masque avec une dilatation légère
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

    # 3. Inpainting (Remplacement des pixels)
    # Remplir les zones du masque avec les couleurs environnantes
    restored_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # Sauvegarder les résultats pour comparer
    cv2.imwrite("etape1_masque_caustiques.jpg", mask)
    cv2.imwrite(output_path, restored_img)
    print(f"Traitement terminé. Résultat sauvegardé dans {output_path}")
    print(f"Regardez 'etape1_masque_caustiques.jpg' pour voir ce que le système a détecté comme reflet.")

if __name__ == "__main__":
    # Test sur une image statique extraite
    # Usage: python test_caustiques_inpaint.py <image_entree> <image_sortie>
    in_img = sys.argv[1] if len(sys.argv) > 1 else 'frame_test.jpg'
    out_img = sys.argv[2] if len(sys.argv) > 2 else 'resultat_sans_caustiques.jpg'
    remove_caustics_single_frame(in_img, out_img)
