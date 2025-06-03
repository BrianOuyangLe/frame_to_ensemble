import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_frame_pairs(video_path, size=(64, 64), max_frames=None):
    """
    Extract consecutive (i, i+1) frame pairs from a video.

    Returns:
        X: np.ndarray of shape (N-1, H, W, 1)   # frame i
        Y: np.ndarray of shape (N-1, H, W, 1)   # frame i+1
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and resize
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()

    frames = np.array(frames)  # (N, H, W)
    frames = np.expand_dims(frames, axis=-1)  # (N, H, W, 1)

    # Make (i, i+1) pairs
    X = frames[:-1]
    Y = frames[1:]

    return X, Y

def show_frame(frame):
    """
    frame: a 2D (H, W) or 3D (H, W, 1) NumPy array with values in [0, 1] or [0, 255]
    """
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame.squeeze(-1)
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.show()
