import cv2
import numpy as np

# Heuristic facial expression classifier (placeholder).
# Uses face bounding box and simple region intensity/variance ratios.
# Categories: Neutral, Positive, Surprised, Frowning

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_expression(frame_bgr):
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 'Unknown'
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        return 'NoFace'
    # Use largest face
    x,y,w,h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_roi = gray[y:y+h, x:x+w]
    if face_roi.size == 0:
        return 'Unknown'
    upper = face_roi[0:int(h*0.4), :]
    mid = face_roi[int(h*0.4):int(h*0.7), :]
    lower = face_roi[int(h*0.7):, :]
    # Features
    var_upper = np.var(upper)
    var_lower = np.var(lower)
    mean_lower = np.mean(lower)
    mean_mid = np.mean(mid)
    mouth_open_score = (mean_lower - mean_mid) / (np.std(lower)+1e-5)
    expressiveness = (var_upper + var_lower) / (np.var(face_roi)+1e-5)
    # Threshold heuristics
    if mouth_open_score > 1.2 and expressiveness > 1.1:
        return 'Surprised'
    if mouth_open_score > 0.6 and expressiveness > 0.9:
        return 'Positive'
    brow_ratio = var_upper / (var_lower + 1e-5)
    if brow_ratio > 1.3 and mouth_open_score < 0.3:
        return 'Frowning'
    return 'Neutral'

def overlay_expression(frame_bgr, label: str):
    out = frame_bgr.copy()
    cv2.putText(out, f"Expr: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return out
