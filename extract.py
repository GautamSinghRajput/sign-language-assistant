"""Step 1: read the dataset and save hand points to data/*.npz

Expected layout (one folder per letter):
    dynamic_folder/ J/ clip1.avi clip2.avi ...   (videos are used; frame photos are skipped)
    static_folder/  A/ img1.jpg img2.jpg ...

Run:
    python extract.py --static path/to/static --dynamic path/to/dynamic
(you can give just one of them)
"""
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from common import (make_hands, get_hands, static_features, raw_frame_features,
                    resample, MIRROR_INPUT)

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv", ".webm")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def files_under(folder, exts):
    found = []
    for root, _, names in os.walk(folder):
        found += [os.path.join(root, n) for n in names if n.lower().endswith(exts)]
    return sorted(found)


def label_folders(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def extract_static(path, max_per_class):
    X, y, skipped = [], [], 0
    with make_hands(static=True) as hands:
        for label in label_folders(path):
            imgs = files_under(os.path.join(path, label), IMAGE_EXT)[:max_per_class]
            for f in tqdm(imgs, desc=f"static {label}"):
                img = cv2.imread(f)
                if img is None:
                    skipped += 1
                    continue
                if MIRROR_INPUT:
                    img = cv2.flip(img, 1)
                found = get_hands(hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                if not found:
                    skipped += 1
                    continue
                X.append(static_features(found[0]))
                y.append(label)
    print(f"Static: kept {len(X)} images, skipped {skipped} (no hand found)")
    return np.array(X), np.array(y)


def extract_dynamic(path, max_per_class):
    X, y = [], []
    with make_hands(static=False) as hands:
        for label in label_folders(path):
            videos = files_under(os.path.join(path, label), VIDEO_EXT)[:max_per_class]
            if not videos:
                print(f"  (no videos in '{label}', skipped)")
            for f in tqdm(videos, desc=f"dynamic {label}"):
                cap = cv2.VideoCapture(f)
                frames = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if MIRROR_INPUT:
                        frame = cv2.flip(frame, 1)
                    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(raw_frame_features(res))
                cap.release()
                if frames:
                    X.append(resample(frames))
                    y.append(label)
    print(f"Dynamic: kept {len(X)} videos")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", help="folder with one sub-folder of photos per letter")
    ap.add_argument("--dynamic", help="folder with one sub-folder of videos per letter")
    ap.add_argument("--max-static", type=int, default=300)
    ap.add_argument("--max-dynamic", type=int, default=100)
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    if a.static:
        X, y = extract_static(a.static, a.max_static)
        np.savez_compressed("data/static.npz", X=X, y=y)
    if a.dynamic:
        X, y = extract_dynamic(a.dynamic, a.max_dynamic)
        np.savez_compressed("data/dynamic.npz", X=X, y=y)
    print("Done. Saved in data/")
