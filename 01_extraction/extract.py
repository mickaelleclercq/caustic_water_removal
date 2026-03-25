import cv2
import os

video_path = "GX010236_synced_enhanced.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 30

# Extract screenshots at 10s, 20s, 30s
times = [10, 20, 30]
for t in times:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"frame_{t}s.jpg", frame)

# Extract a small subclip of 5 seconds (5s to 10s)
start_time = 5
end_time = 10
cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('subclip.mp4', fourcc, fps, (width, height))

current_time = start_time
while current_time <= end_time:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    current_time += 1 / fps

cap.release()
out.release()
print("Extraction complete.")
