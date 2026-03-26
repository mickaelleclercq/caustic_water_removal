"""
Précomputation partagée des homographies SIFT pour tous les scripts GPU.

Calcule les paires (i, j) pour half = 4 (max de toutes les méthodes : N≤9),
ce qui couvre :
  Méthode A (half=2), D (half=4), E (half=3), J (half=4)

Sauvegarde : homography_cache_half4.pkl (quelques Mo, chargement instantané)
Durée      : ~3.5 min (16 workers CPU) pour 1161 frames × 8 paires = 9268 SIFT

Usage :
  python precompute_homographies.py
"""
import cv2
import numpy as np
import time
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

VIDEO_PATH  = os.path.join(os.path.dirname(__file__), 'GX010236_synced_enhanced.MP4')
CACHE_PATH  = os.path.join(os.path.dirname(__file__), 'homography_cache_half4.pkl')
SIFT_SCALE  = 0.25
MAX_HALF    = 4   # couvre N=5 (half=2), N=7 (half=3), N=9 (half=4)
N_WORKERS   = 16


def load_small_frames(video_path, scale):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (0, 0), fx=scale, fy=scale))
    cap.release()
    return frames, fps


def compute_sift_homography(ref_small, neighbor_small):
    sift = cv2.SIFT_create(nfeatures=2000)
    g1   = cv2.cvtColor(ref_small,      cv2.COLOR_BGR2GRAY)
    g2   = cv2.cvtColor(neighbor_small, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    flann   = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    return H


def main():
    if os.path.exists(CACHE_PATH):
        print(f"Cache déjà disponible : {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        print(f"  {len(cache)} homographies chargées.")
        return

    print(f"=== Précomputation homographies (half={MAX_HALF}) ===")
    t0 = time.time()

    print("Chargement frames small (0.25×)…")
    frames, fps = load_small_frames(VIDEO_PATH, SIFT_SCALE)
    n = len(frames)
    print(f"  {n} frames en {time.time()-t0:.1f}s")

    tasks = [(i, j)
             for i in range(n)
             for j in range(max(0, i - MAX_HALF), min(n, i + MAX_HALF + 1))
             if j != i]
    print(f"  {len(tasks)} paires à calculer ({N_WORKERS} workers)…")

    H_cache = {}
    done    = 0
    t1      = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(compute_sift_homography, frames[i], frames[j]): (i, j)
                for i, j in tasks}
        for fut in as_completed(futs):
            i, j = futs[fut]
            H_cache[(i, j)] = fut.result()
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t1
                eta     = elapsed / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    total = time.time() - t0
    print(f"\n  {len(H_cache)} homographies calculées en {total:.1f}s")

    print(f"Sauvegarde → {CACHE_PATH}")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(H_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(CACHE_PATH) / 1e6
    print(f"  Cache sauvegardé ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
