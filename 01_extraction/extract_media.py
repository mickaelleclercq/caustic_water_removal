import os
from moviepy.editor import VideoFileClip
from PIL import Image

def main():
    video_path = "GX010236_synced_enhanced.MP4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    print("Loading video...")
    clip = VideoFileClip(video_path)
    
    video_duration = clip.duration
    print(f"Video duration is: {video_duration} seconds.")
    
    # Extract screenshots
    times_to_extract = [5.0, 15.0, 30.0]
    for t in times_to_extract:
        if t < video_duration:
            print(f"Extracting frame at {t}s...")
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            img.save(f"screenshot_{int(t)}s.jpg")
        else:
            print(f"Time {t}s is beyond video duration ({video_duration}s).")
            
    # Extract subclip
    start_time = 10.0
    end_time = 15.0
    
    if start_time < video_duration:
        end_time = min(end_time, video_duration)
        print(f"Extracting subclip from {start_time}s to {end_time}s...")
        subclip = clip.subclip(start_time, end_time)
        subclip.write_videofile("subclip_5s.mp4", codec="libx264", audio_codec="aac")
        print("Subclip saved as subclip_5s.mp4")
    else:
        print("Video is too short for this subclip extraction.")
        
    clip.close()

if __name__ == "__main__":
    main()
