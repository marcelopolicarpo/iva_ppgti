import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

mp_pose = mp.solutions.pose

def calcular_angulo(p1, p2, p3):
    """Calcula o ângulo entre três pontos (em graus)."""
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    angulo = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) -
                        np.arctan2(a[1]-b[1], a[0]-b[0]))
    angulo = abs(angulo)
    if angulo > 180:
        angulo = 360 - angulo
    return angulo


# === Captura da Webcam ===
cap = cv2.VideoCapture(0)

# Define o tamanho desejado da janela (largura x altura)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dados para salvar
tempos, angulos_esq, angulos_dir = [], [], []

# Gráfico dinâmico
plt.ion()
fig, ax = plt.subplots()
linha_esq, = ax.plot([], [], label='Braço Esquerdo')
linha_dir, = ax.plot([], [], label='Braço Direito')
ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Ângulo (°)')
ax.set_ylim(0, 180)
ax.legend()

inicio = time.time()

# Pose do MediaPipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensiona frame (opcional)
        frame = cv2.resize(frame, (1280, 720))

        # Processa imagem
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = frame.copy()

        try:
            lm = results.pose_landmarks.landmark

            # Braço esquerdo
            ombro_esq = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            cotovelo_esq = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            quadril_esq = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            angulo_esq = calcular_angulo(quadril_esq, ombro_esq, cotovelo_esq)

            # Braço direito
            ombro_dir = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            cotovelo_dir = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            quadril_dir = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            angulo_dir = calcular_angulo(quadril_dir, ombro_dir, cotovelo_dir)

            # Converte coordenadas normalizadas para pixels
            def to_pixel(p): 
                return tuple(np.multiply(p, [image.shape[1], image.shape[0]]).astype(int))

            # Desenha apenas os segmentos desejados
            for (a, b, c, cor) in [
                (quadril_esq, ombro_esq, cotovelo_esq, (0, 255, 0)),
                (quadril_dir, ombro_dir, cotovelo_dir, (0, 255, 255))
            ]:
                cv2.line(image, to_pixel(a), to_pixel(b), cor, 3)
                cv2.line(image, to_pixel(b), to_pixel(c), cor, 3)
                cv2.circle(image, to_pixel(a), 8, cor, -1)
                cv2.circle(image, to_pixel(b), 8, cor, -1)
                cv2.circle(image, to_pixel(c), 8, cor, -1)

            # Mostra ângulos
            cv2.putText(image, f"Esq: {int(angulo_esq)}",
                        to_pixel(ombro_esq),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(image, f"Dir: {int(angulo_dir)}",
                        to_pixel(ombro_dir),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Atualiza dados
            tempo = time.time() - inicio
            tempos.append(tempo)
            angulos_esq.append(angulo_esq)
            angulos_dir.append(angulo_dir)

            # Atualiza gráfico
            linha_esq.set_data(tempos, angulos_esq)
            linha_dir.set_data(tempos, angulos_dir)
            ax.set_xlim(max(0, tempo - 30), tempo + 1)
            plt.pause(0.01)

        except:
            pass

        # Exibe vídeo em tamanho grande
        cv2.imshow("Angulo dos Braços - MediaPipe Pose (1280x720)", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# === Salva CSV ===
dados = pd.DataFrame({
    "tempo_segundos": tempos,
    "angulo_braco_esquerdo": angulos_esq,
    "angulo_braco_direito": angulos_dir
})
dados.to_csv("angulos_bracos.csv", index=False)
print("\n✅ Dados salvos em 'angulos_bracos.csv'")
