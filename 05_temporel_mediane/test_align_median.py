import cv2
import numpy as np
import time

def get_frames(video_path, center_sec, count=7, step=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    center_frame = int(center_sec * fps)
    
    start_frame = center_frame - (count // 2) * step
    
    frames = []
    for i in range(count):
        idx = start_frame + i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def align_images(frames):
    center_idx = len(frames) // 2
    ref_frame = frames[center_idx]
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    
    aligned_frames = [None] * len(frames)
    aligned_frames[center_idx] = ref_frame
    
    # Create ORB detector to align backgrounds (sand, stable rocks)
    orb = cv2.ORB_create(nfeatures=5000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    
    for i, frame in enumerate(frames):
        if i == center_idx: 
            continue
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_cur, des_cur = orb.detectAndCompute(frame_gray, None)
        
        matches = matcher.match(des_cur, des_ref, None)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:500]
        
        src_pts = np.float32([kp_cur[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = ref_frame.shape[:2]
        aligned_frame = cv2.warpPerspective(frame, M, (w, h))
        aligned_frames[i] = aligned_frame
        
    return aligned_frames, frames[center_idx]

def main():
    video_path = "subclip_5s.mp4"
    print("Extraction de 7 frames autour de la 2.5ème seconde...")
    frames = get_frames(video_path, center_sec=2.5, count=7, step=3)
    if len(frames) < 3:
        print("Erreur: pas assez de frames.")
        return
        
    print("Alignement de la caméra (compensation du mouvement)...")
    aligned_frames, original = align_images(frames)
    
    print("Calcul de la Médiane Temporelle...")
    # La médiane va conserver le fond net (aligné) et jeter les aberrations lumineuses
    # mouvantes (caustiques) qui changent à chaque frame.
    median_frame = np.median(aligned_frames, axis=0).astype(np.uint8)
    
    cv2.imwrite("comparaison_1_originale.jpg", original)
    cv2.imwrite("comparaison_2_mediane_alignee.jpg", median_frame)
    print("Terminé ! 'comparaison_1_originale.jpg' et 'comparaison_2_mediane_alignee.jpg' créés.")

if __name__ == "__main__":
    main()
