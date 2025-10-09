import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

import numpy as np
import math
import mediapipe as mp

mp_pose = mp.solutions.pose

# Dicionário global para armazenar o maior ângulo de cada braço
max_arm_angle = {"Left": 0.0, "Right": 0.0}

def angulo_abertura_braco(pose_landmarks, lado):
    """
    Calcula a abertura do braço (ângulo entre o tronco e o braço) para o lado especificado.

    Parâmetros:
        pose_landmarks: landmarks detectados pelo MediaPipe Pose (results.pose_landmarks.landmark)
        lado: 'Left' ou 'Right' (braço esquerdo ou direito)

    Retorna:
        ângulo em graus entre o braço e o tronco
    """
    global max_arm_angle

    if lado == "Left":
        ombro = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        cotovelo = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                             pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y])
        quadril = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                            pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
    else:
        ombro = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        cotovelo = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                             pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y])
        quadril = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                            pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])

    # Vetor do tronco (quadril -> ombro)
    tronco = ombro - quadril

    # Vetor do braço (ombro -> cotovelo)
    braco = cotovelo - ombro

    # Produto escalar e normas
    dot = np.dot(tronco, braco)
    norm_product = np.linalg.norm(tronco) * np.linalg.norm(braco)

    if norm_product == 0:
        angulo = 0.0
    else:
        angulo = math.acos(np.clip(dot / norm_product, -1.0, 1.0))
        angulo = math.degrees(angulo)

    # Atualiza ângulo máximo registrado
    if angulo > max_arm_angle[lado]:
        max_arm_angle[lado] = angulo

    return angulo


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        ang_esq = angulo_abertura_braco(results.pose_landmarks.landmark, "Left")
        ang_dir = angulo_abertura_braco(results.pose_landmarks.landmark, "Right")
        print(f"Ângulo braço esquerdo: {ang_esq:.1f}° | direito: {ang_dir:.1f}°")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
