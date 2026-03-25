import cv2
import numpy as np

# Créez une préview réduite de chaque screenshot
for t in [5, 15, 30]:
    img = cv2.imread(f"screenshot_{t}s.jpg")
    if img is not None:
        small = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
        cv2.imwrite(f"preview_{t}s.jpg", small)

print("Done")
