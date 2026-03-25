import cv2
import os

def main():
    video_path = "GX010236_synced_enhanced.MP4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened. Total frames: {total_frames}, FPS: {fps}")

    # Extract screenshots
    times_to_extract = [5, 15, 30]
    for t in times_to_extract:
        frame_idx = int(t * fps)
        if frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"screenshot_{t}s.jpg", frame)
                print(f"Saved screenshot_{t}s.jpg")

    # Extract a 5-sec subclip (from 10s to 15s)
    start_sec = 10
    end_sec = 15
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('subclip_5s.mp4', fourcc, fps, (width, height))
    
    print(f"Extracting subclip from {start_sec}s to {end_sec}s...")
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print("Done extracting. Info:")
    os.system("ls -lh screenshot*.jpg subclip_5s.mp4")

if __name__ == "__main__":
    main()
