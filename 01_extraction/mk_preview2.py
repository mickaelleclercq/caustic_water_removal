import cv2
import numpy as np

for t in [5, 15, 30]:
    img = cv2.imread(f"result_msrcp_{t}s.jpg")
    if img is not None:
        small = cv2.resize(img, (0,0), fx=0.35, fy=0.35)
        cv2.imwrite(f"preview_msrcp_{t}s.jpg", small)
print("Done")
