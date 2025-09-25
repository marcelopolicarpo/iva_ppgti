import cv2
import mediapipe as mp
import pygame
import os

# Inicializa o mixer do pygame
pygame.mixer.init()

# Carrega os sons da pasta "sons"
sons = [pygame.mixer.Sound(os.path.join("sons", f"som{i}.wav")) for i in range(1, 11)]

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Captura de vídeo
cap = cv2.VideoCapture(0)

# IDs dos pontos finais dos dedos
ponta_dedos_ids = [4, 8, 12, 16, 20]

# Estado anterior dos dedos
dedos_anteriores = [0] * 10  # 10 dedos (5 esquerda + 5 direita)

def detectar_dedos(hand_landmarks, hand_label):
    """Retorna lista com 5 valores (0 ou 1) indicando se cada dedo está levantado"""
    dedos = []

    # Polegar
    if hand_label == "Right":
        dedos.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:  # Left
        dedos.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    # Outros dedos
    for id in range(1, 5):
        if hand_landmarks.landmark[ponta_dedos_ids[id]].y < hand_landmarks.landmark[ponta_dedos_ids[id] - 2].y:
            dedos.append(1)
        else:
            dedos.append(0)

    return dedos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dedos = detectar_dedos(hand_landmarks, hand_label)
            offset = 0 if hand_label == "Left" else 5

            for i, dedo in enumerate(dedos):
                pos = offset + i
                if dedo == 1 and dedos_anteriores[pos] == 0:
                    sons[pos].play()
                dedos_anteriores[pos] = dedo

    cv2.imshow("Piano com as Maos", frame)

    # Fecha com Q ou botão X
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Piano com as Maos", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
