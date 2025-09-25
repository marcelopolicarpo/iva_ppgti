# Requisitos: pip install mediapipe opencv-python numpy
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def angle_between_points(a, b, c):
    # calcula ângulo ABC em graus
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang

cap = cv2.VideoCapture(0)
# buffer para suavizar ângulo
angle_buffer = deque(maxlen=5)

# Exemplo: queremos que o cotovelo direito dobre até ~45 graus (target) com tolerância +/-10
TARGET_ANGLE = 45.0
TOLERANCE = 10.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    feedback_text = "Sem detecção"
    color = (0, 0, 255)  # vermelho por padrão

    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark

        # landmarks: RIGHT_SHOULDER = 12, RIGHT_ELBOW = 14, RIGHT_WRIST = 16 (MediaPipe indexing)
        shoulder = (lm[12].x * w, lm[12].y * h)
        elbow = (lm[14].x * w, lm[14].y * h)
        wrist   = (lm[16].x * w, lm[16].y * h)

        ang = angle_between_points(shoulder, elbow, wrist)
        angle_buffer.append(ang)
        smooth_ang = np.mean(angle_buffer)

        # lógica simples de correção
        diff = abs(smooth_ang - TARGET_ANGLE)
        if diff <= TOLERANCE:
            feedback_text = f"OK (ângulo: {smooth_ang:.1f}°)"
            color = (0, 255, 0)  # verde
        elif smooth_ang < TARGET_ANGLE - TOLERANCE:
            feedback_text = f"Aumentar flexão (ângulo: {smooth_ang:.1f}°)"
            color = (0, 165, 255)  # laranja
        else:
            feedback_text = f"Reduzir flexão (ângulo: {smooth_ang:.1f}°)"
            color = (0, 165, 255)

        # desenha esqueleto simples e ângulo
        cv2.circle(frame, tuple(map(int, shoulder)), 5, color, -1)
        cv2.circle(frame, tuple(map(int, elbow)), 5, color, -1)
        cv2.circle(frame, tuple(map(int, wrist)), 5, color, -1)
        cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, elbow)), color, 2)
        cv2.line(frame, tuple(map(int, elbow)), tuple(map(int, wrist)), color, 2)
        cv2.putText(frame, f"{smooth_ang:.1f} deg", (int(elbow[0]+10), int(elbow[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Rehab Pose Checker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
