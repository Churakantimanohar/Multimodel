import cv2
import numpy as np
from typing import Dict

try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    _HAS_MP = True
except Exception:
    mp_face = None
    _HAS_MP = False


def extract_face_landmarks(frame) -> np.ndarray:
    if not _HAS_MP or mp_face is None:
        # Fallback: return zero landmark set
        return np.zeros((1, 468, 3), dtype=np.float32)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)
    if not result.multi_face_landmarks:
        return np.zeros((1, 468, 3), dtype=np.float32)
    landmarks = result.multi_face_landmarks[0].landmark
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    return arr.reshape(1, 468, 3)


def process_video(path: str, max_frames: int = 64) -> Dict[str, np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames_landmarks = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        lm = extract_face_landmarks(frame)
        frames_landmarks.append(lm)
        count += 1
    cap.release()
    if not frames_landmarks:
        return {'landmarks': np.zeros((1,468,3), dtype=np.float32)}
    return {'landmarks': np.concatenate(frames_landmarks, axis=0)}
