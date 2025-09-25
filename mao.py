import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def contar_dedos(hand_landmarks, hand_label):
    dedos = []
    ponta_dedos_ids = [4, 8, 12, 16, 20]

    # Polegar (x) depende da mão
    if hand_label == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            dedos.append(1)
        else:
            dedos.append(0)
    else:  # mão esquerda
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            dedos.append(1)
        else:
            dedos.append(0)

    # Outros dedos (y)
    for id in range(1,5):
        if hand_landmarks.landmark[ponta_dedos_ids[id]].y < hand_landmarks.landmark[ponta_dedos_ids[id]-2].y:
            dedos.append(1)
        else:
            dedos.append(0)

    return sum(dedos)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Identifica se é mão direita ou esquerda
            hand_label = results.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            numero_dedos = contar_dedos(hand_landmarks, hand_label)

            cv2.putText(frame, f'{hand_label} Mao: {numero_dedos}', (50, 50 + idx*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Contador de Dedos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Contador de Dedos", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
