import cv2
import numpy as np
import time

input_vid = 'subclip_5s.mp4'
output_vid = 'subclip_5s_1080p_morph_test.mp4'

cap = cv2.VideoCapture(input_vid)
fps = cap.get(cv2.CAP_PROP_FPS)

# Original resolution
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# We will scale down by 2 (to 1080p roughly) for a much faster test.
scale = 0.5
w = int(orig_w * scale)
h = int(orig_h * scale)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid, fourcc, fps, (w, h))

if not cap.isOpened():
    print("Error opening video")
    exit()

print(f"Starting Morphological test on {total_frames} frames.")
print(f"Resizing from {orig_w}x{orig_h} to {w}x{h} for faster processing...")

kernel_size = 9 # smaller kernel since video is scaled down
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize frame
    frame = cv2.resize(frame, (w, h))
        
    # 1. Convert to HSV and extract Value channel (brightness)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # 2. Top-Hat morphological transform to isolate bright peak details
    tophat = cv2.morphologyEx(v_channel, cv2.MORPH_TOPHAT, kernel)
    
    # 3. Threshold to get a binary mask of the caustics
    _, mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    
    # 4. Inpaint the masked regions to blend with surroundings
    restored = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    
    out.write(restored)
    frame_count += 1
    
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_proc = frame_count / elapsed
        print(f"Processed {frame_count}/{total_frames} frames... ({fps_proc:.2f} fps)")

out.release()
cap.release()
print(f"Test finished! Took {time.time() - start_time:.2f} seconds.")
print(f"Output saved to {output_vid}")
