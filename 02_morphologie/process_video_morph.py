import cv2
import numpy as np
import time
import os

input_vid = 'subclip_5s.mp4'
output_vid = 'subclip_5s_morph.mp4'

cap = cv2.VideoCapture(input_vid)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid, fourcc, fps, (w, h))

if not cap.isOpened():
    print("Error opening video")
    exit()

print(f"Starting Morphological Inpainting test on {total_frames} frames at {w}x{h} resolution...")

# We use an elliptical kernel size relative to the high resolution of the video
# Assuming the video is 4K+, a kernel of 15-21 is usually good to catch the "strings" of light
kernel_size = max(15, int(w / 150))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

# To avoid endless processing if it's too slow on CPU, we will process only the first 60 frames (2 seconds)
# for this quick test. You can process the whole clip later.
frames_to_process = min(total_frames, 60)

start_time = time.time()
frame_count = 0

while frame_count < frames_to_process:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 1. Convert to HSV and extract Value channel (brightness)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # 2. Top-Hat morphological transform to isolate bright peak details
    tophat = cv2.morphologyEx(v_channel, cv2.MORPH_TOPHAT, kernel)
    
    # 3. Threshold to get a binary mask of the caustics
    # Parameter 50 is a mid-range threshold. 
    _, mask = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)
    
    # Slightly dilate mask to cover edges of the light lines
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    
    # 4. Inpaint the masked regions to blend with surroundings
    # INPAINT_TELEA is faster than NS. Radius 3.
    restored = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    
    out.write(restored)
    frame_count += 1
    
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps_proc = frame_count / elapsed
        print(f"Processed {frame_count}/{frames_to_process} frames... ({fps_proc:.2f} frames/sec)")

out.release()
cap.release()
print(f"Test finished! Took {time.time() - start_time:.2f} seconds.")
print(f"Output saved to {output_vid}")
