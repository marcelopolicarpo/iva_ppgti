import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
import datetime
import pygame  # Adicionei esta importação para o som

# --- Inicialização do Pygame Mixer para Áudio ---
pygame.mixer.init()

# Carrega o efeito sonoro de faca cortando
try:
    knife_sound = pygame.mixer.Sound(r"./knife-cut.mp3")  # Ajuste o caminho conforme necessário
    print("✅ Efeito sonoro carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar efeito sonoro: {e}")
    knife_sound = None

# --- Configurações Iniciais ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


# Aumenta o tamanho da captura de vídeo (se a câmera suportar)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# IDs dos pontos de referência da mão
INDEX_FINGER_TIP_LM = mp_hands.HandLandmark.INDEX_FINGER_TIP
WRIST_LM = mp_hands.HandLandmark.WRIST

# --- Propriedades das Frutas (Cores Aprimoradas) ---
FRUIT_PROPERTIES = [
    {
        'id': 'apple', 'radius': 40, 'speed_y_range': (-25, -20),
        'colors': {
            'base': (0, 0, 200), 'dark': (0, 0, 120), 'light': (80, 80, 255),
            'highlight': (220, 220, 255), 'stem_color': (20, 70, 30), 'leaf_color': (0, 180, 0)
        }, 'shine_intensity': 0.8
    },
    {
        'id': 'watermelon', 'radius': 55, 'speed_y_range': (-30, -25),
        'colors': {
            'base': (0, 150, 0), 'dark': (0, 80, 0), 'light': (50, 200, 50),
            'inner': (50, 50, 220), 'seed': (10, 10, 10), 'stripe_color': (0, 100, 0),
            'rind': (210, 230, 210)  # Cor da casca blanco interna
        }, 'shine_intensity': 0.5
    },
    {
        'id': 'orange', 'radius': 45, 'speed_y_range': (-22, -18),
        'colors': {
            'base': (0, 140, 255), 'dark': (0, 100, 200), 'light': (50, 180, 255),
            'highlight': (180, 220, 255), 'texture_dots_color': (0, 120, 230)
        }, 'shine_intensity': 0.7
    },
    {
        'id': 'banana', 'radius': 30, 'speed_y_range': (-24, -19),
        'colors': {
            'base': (0, 215, 255), 'dark': (0, 150, 200), 'light': (100, 255, 255),
            'tip': (10, 40, 70), 'stem_color': (30, 60, 90)
        }, 'shine_intensity': 0.6
    },
    {
        'id': 'bomb', 'radius': 45, 'speed_y_range': (-20, -15),
        'colors': {
            'base': (60, 60, 60), 'dark': (20, 20, 20), 'light': (100, 100, 100),
            'highlight': (200, 200, 200), 'fuse_color': (200, 180, 150), 'spark_color': (0, 255, 255)
        }, 'shine_intensity': 0.9
    },
]

# --- Variáveis Globais do Jogo ---
fruits_on_screen = []
last_spawn_time = time.time()
SPAWN_INTERVAL = 1.2
MAX_FRUITS_ON_SCREEN = 3
SCORE_RIGHT = 0  # Pontuação da mão direita
SCORE_LEFT = 0   # Pontuação da mão esquerda
start_time = time.time()  # Tempo de início do jogo
previous_tip_positions = {}
blade_trail = []

# --- Variáveis para Avaliações Biomecânicas ---
prev_time = time.time()
prev_wrist_positions = {'left': None, 'right': None}
hand_speeds = {'left': 0.0, 'right': 0.0}  # Velocidade em pixels/segundo
moving_times = {'left': 0.0, 'right': 0.0}  # Tempo total em movimento (segundos)
arm_openings = {'left': 0.0, 'right': 0.0}  # Ângulo de abertura do braço (graus)
MOVING_SPEED_THRESHOLD = 50  # Pixels/segundo para considerar "em movimento"

# --- Histórico para relatório clínico ---
arm_opening_history = {'left': [], 'right': []}
hand_speed_history = {'left': [], 'right': []}

# --- Funções do Jogo (Lógica) ---

def spawn_fruit(frame_w, frame_h):
    if len(fruits_on_screen) >= MAX_FRUITS_ON_SCREEN:
        return
    prop = random.choice(FRUIT_PROPERTIES)
    x = random.randint(prop['radius'], frame_w - prop['radius'])
    y = frame_h + prop['radius']
    speed_x = random.uniform(-3, 3)
    speed_y = random.uniform(*prop['speed_y_range'])
    gravity = 0.5
    fruits_on_screen.append({
        'prop': prop, 'x': x, 'y': y, 'vx': speed_x, 'vy': speed_y,
        'gravity': gravity, 'status': 'active', 'cut_angle': 0,
        'cut_impulse_x': 0, 'cut_impulse_y': 0
    })

def update_fruit_position(fruit):
    if fruit['status'] == 'active':
        fruit['y'] += fruit['vy']
        fruit['x'] += fruit['vx']
        fruit['vy'] += fruit['gravity']
    elif fruit['status'] == 'cut':
        fruit['x'] += fruit['vx'] + fruit['cut_impulse_x']
        fruit['y'] += fruit['vy'] + fruit['cut_impulse_y']
        fruit['cut_impulse_x'] *= 0.95
        fruit['cut_impulse_y'] *= 0.95
        fruit['vy'] += fruit['gravity']

def check_collision_and_cut(fruit, current_tip_x, current_tip_y, prev_tip_x, prev_tip_y, is_right_hand):
    global SCORE_RIGHT, SCORE_LEFT
    if fruit['status'] != 'active':
        return False

    fruit_center = np.array([fruit['x'], fruit['y']])
    p1 = np.array([prev_tip_x, prev_tip_y])
    p2 = np.array([current_tip_x, current_tip_y])

    line_vec = p2 - p1
    fruit_vec = fruit_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        distance = np.linalg.norm(fruit_center - p1)
    else:
        t = max(0, min(1, np.dot(fruit_vec, line_vec) / line_len_sq))
        closest_point_on_line = p1 + t * line_vec
        distance = np.linalg.norm(fruit_center - closest_point_on_line)

    if distance < fruit['prop']['radius']:
        if fruit['prop']['id'] == 'bomb':
            if is_right_hand:
                SCORE_RIGHT = 0
            else:
                SCORE_LEFT = 0
            print("❌ Tocou na BOMBA! Pontuação zerada.")
        else:
            if is_right_hand:
                SCORE_RIGHT += 1
            else:
                SCORE_LEFT += 1
            print(f"✔️ Corte! Pontuação - Direita: {SCORE_RIGHT}, Esquerda: {SCORE_LEFT}")
            
            # Toca o efeito sonoro quando corta uma fruta (exceto bomba)
            if knife_sound is not None:
                knife_sound.play()
                print("🔊 Som de corte reproduzido!")

        fruit['status'] = 'cut'
        fruit['cut_angle'] = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.pi / 2
        fruit['vy'] -= 8
        cut_strength = 6
        fruit['cut_impulse_x'] = math.cos(fruit['cut_angle']) * cut_strength
        fruit['cut_impulse_y'] = math.sin(fruit['cut_angle']) * cut_strength
        return True
    return False

# --- Funções de Desenho 3D (Otimizadas) ---

def draw_3d_sphere(frame, center, radius, colors, shine_intensity):
    x, y = center
    for i in range(radius, 0, -3):
        ratio = i / radius
        interp_color = tuple(int(dc + (lc - dc) * ratio) for dc, lc in zip(colors['dark'], colors['light']))
        cv2.circle(frame, (x, y), i, interp_color, -1, lineType=cv2.LINE_AA)

def draw_3d_cut_fruit(frame, fruit):
    x, y, r = int(fruit['x']), int(fruit['y']), fruit['prop']['radius']
    colors = fruit['prop']['colors']
    angle_rad, half_r, separation = fruit['cut_angle'], int(r * 0.8), 15
    dx, dy = math.cos(angle_rad) * separation, math.sin(angle_rad) * separation
    x1, y1 = int(x + dx), int(y + dy)
    x2, y2 = int(x - dx), int(y - dy)

    inner_color = colors.get('inner', colors['base'])

    # Metades da polpa
    cv2.circle(frame, (x1, y1), half_r, inner_color, -1)
    cv2.circle(frame, (x2, y2), half_r, inner_color, -1)

    # Detalhes da fruta cortada
    if fruit['prop']['id'] == 'watermelon':
        rind_thickness = int(half_r * 0.1)
        cv2.circle(frame, (x1, y1), half_r, colors['rind'], rind_thickness)
        cv2.circle(frame, (x2, y2), half_r, colors['rind'], rind_thickness)
        # Sementes
        for i in range(3):
            angle = random.uniform(0, math.pi * 2)
            dist = random.uniform(0, half_r * 0.7)
            cv2.circle(frame, (int(x1 + dist * math.cos(angle)), int(y1 + dist * math.sin(angle))), 2, colors['seed'], -1)
            cv2.circle(frame, (int(x2 + dist * math.cos(angle)), int(y2 + dist * math.sin(angle))), 2, colors['seed'], -1)

    elif fruit['prop']['id'] == 'orange':
        # Linhas dos gomos
        for i in range(8):
            angle = i * (math.pi / 4)
            cv2.line(frame, (x1, y1), (int(x1 + math.cos(angle) * half_r), int(y1 + math.sin(angle) * half_r)), colors['highlight'], 1)
            cv2.line(frame, (x2, y2), (int(x2 + math.cos(angle) * half_r), int(y2 + math.sin(angle) * half_r)), colors['highlight'], 1)

    angle_deg = math.degrees(angle_rad)
    cv2.ellipse(frame, (x1, y1), (half_r, half_r), angle_deg, 90, 270, colors['base'], -1)
    cv2.ellipse(frame, (x2, y2), (half_r, half_r), angle_deg, -90, 90, colors['base'], -1)

def draw_fruit_3d(frame, fruit):
    x, y, r = int(fruit['x']), int(fruit['y']), fruit['prop']['radius']
    colors = fruit['prop']['colors']

    if fruit['status'] == 'active':
        draw_3d_sphere(frame, (x, y), r, colors, fruit['prop']['shine_intensity'])

        if fruit['prop']['id'] == 'apple':
            cv2.line(frame, (x, y - r + 5), (x + 5, y - r - int(r * 0.4)), colors['stem_color'], 3)
            cv2.ellipse(frame, (x + 10, y - r - int(r * 0.3)), (int(r * 0.25), int(r * 0.1)), 45, 0, 360, colors['leaf_color'], -1)
            cv2.circle(frame, (x - r // 3, y - r // 3), r // 6, colors['highlight'], -1, lineType=cv2.LINE_AA)

        elif fruit['prop']['id'] == 'watermelon':
            for i in range(5):
                angle = i * 72
                pts = []
                for j in range(20):
                    a = math.radians(angle + random.uniform(-5, 5))
                    rad = r * (1 - (j / 20) ** 2 * 0.5 + random.uniform(-0.05, 0.05))
                    px = int(x + rad * math.sin(a) * math.cos(math.radians(j * 10)))
                    py = int(y - rad * math.cos(a))
                    if j > 0:
                        cv2.line(frame, (pts[-1][0], pts[-1][1]), (px, py), colors['stripe_color'], int(r * 0.08))
                    pts.append([px, py])

        elif fruit['prop']['id'] == 'orange':
            for _ in range(30):
                angle, dist = random.uniform(0, 2 * math.pi), random.uniform(0, r * 0.9)
                px, py = int(x + dist * math.cos(angle)), int(y + dist * math.sin(angle))
                cv2.circle(frame, (px, py), 1, colors['texture_dots_color'], -1)
            cv2.circle(frame, (x, y - r + 5), 3, colors['dark'], -1)  # "Umbigo"
            cv2.circle(frame, (x - r // 3, y - r // 3), r // 5, colors['highlight'], -1, lineType=cv2.LINE_AA)

        elif fruit['prop']['id'] == 'bomb':
            fuse_end = (x + r // 2, y - r - r // 4)
            cv2.line(frame, (x, y - r), fuse_end, colors['fuse_color'], 3)
            if time.time() % 0.2 < 0.1:
                cv2.circle(frame, fuse_end, 7, colors['spark_color'], -1)
            cv2.circle(frame, (x - r // 3, y - r // 3), r // 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    elif fruit['status'] == 'cut':
        draw_3d_cut_fruit(frame, fruit)

# --- Funções para Avaliações Biomecânicas ---

def calculate_hand_speed(hand_landmarks, hand_label, frame_w, frame_h, delta_time):
    """
    Calcula a velocidade da mão com base na posição do pulso.
    Velocidade em pixels por segundo.
    """
    global prev_wrist_positions, hand_speeds
    wrist_lm = hand_landmarks.landmark[WRIST_LM]
    current_wrist = np.array([wrist_lm.x * frame_w, wrist_lm.y * frame_h])
    
    side = 'right' if hand_label == 'Right' else 'left'
    prev_pos = prev_wrist_positions[side]
    
    if prev_pos is not None:
        distance = np.linalg.norm(current_wrist - prev_pos)
        speed = distance / delta_time if delta_time > 0 else 0
        hand_speeds[side] = speed  # Atualiza velocidade atual
        if speed > 0:
            hand_speed_history[side].append(speed)

    else:
        hand_speeds[side] = 0.0
    
    prev_wrist_positions[side] = current_wrist
    
    return hand_speeds[side]

def calculate_moving_time(hand_speed, side, delta_time):
    """
    Acumula o tempo que a mão passa em movimento se a velocidade exceder o threshold.
    """
    global moving_times
    if hand_speed > MOVING_SPEED_THRESHOLD:
        moving_times[side] += delta_time

def calculate_arm_opening(pose_landmarks, frame_w, frame_h, side):
    global arm_openings

    if side == 'right':
        shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    else:
        shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    if shoulder.visibility > 0.5 and wrist.visibility > 0.5:
        shoulder_pos = np.array([shoulder.x * frame_w, shoulder.y * frame_h])
        wrist_pos = np.array([wrist.x * frame_w, wrist.y * frame_h])

        arm_vector = wrist_pos - shoulder_pos
        if np.linalg.norm(arm_vector) == 0:
            return 0.0

        arm_norm = arm_vector / np.linalg.norm(arm_vector)
        vertical_vector = np.array([0, 1])

        dot_product = np.dot(arm_norm, vertical_vector)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        arm_openings[side] = angle_deg
        return angle_deg

    arm_openings[side] = 0.0
    return 0.0


#salvando dados clínicos em CSV para análise posterior
import csv
from datetime import datetime

def save_game_metrics_to_csv():
    session_time = time.time() - start_time

    def safe_mean(data):
        return sum(data) / len(data) if data else 0.0

    def safe_max(data):
        return max(data) if data else 0.0

    row = {
        "Jogador": player_name,
        "Data_Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Duracao_Sessao_segundos": round(session_time, 2),

        "Pontuacao_Mao_Direita": SCORE_RIGHT,
        "Pontuacao_Mao_Esquerda": SCORE_LEFT,

        "Velocidade_Media_Mao_Direita_px_s": round(safe_mean(hand_speed_history['right']), 2),
        "Velocidade_Media_Mao_Esquerda_px_s": round(safe_mean(hand_speed_history['left']), 2),

        "Tempo_Movimento_Mao_Direita_segundos": round(moving_times['right'], 2),
        "Tempo_Movimento_Mao_Esquerda_segundos": round(moving_times['left'], 2),

        "Abertura_Media_Braco_Direito_graus": round(safe_mean(arm_opening_history['right']), 2),
        "Abertura_Media_Braco_Esquerdo_graus": round(safe_mean(arm_opening_history['left']), 2),

        "Abertura_Maxima_Braco_Direito_graus": round(safe_max(arm_opening_history['right']), 2),
        "Abertura_Maxima_Braco_Esquerdo_graus": round(safe_max(arm_opening_history['left']), 2),
    }

    filename = "game_clinical_metrics_new.csv"
    file_exists = False

    try:
        with open(filename, "r"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print("📁 Métricas salvas com sucesso em:", filename)



# --- Loop Principal ---

# --- Nome do jogador (antes de iniciar o jogo) ---
player_name = input("Digite o nome do jogador: ")
start_time = time.time()  # Reinicia o tempo após inserir o nome

cv2.namedWindow("Fruit Ninja 3D - Duas Mãos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fruit Ninja 3D - Duas Mãos", 600, 400)  # Aumenta o tamanho da janela

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb_frame)
    results_pose = pose.process(rgb_frame)

    #frame[:] = (50, 40, 30)

    current_time = time.time()
    delta_time = current_time - prev_time
    prev_time = current_time

    # Rastreamento e Desenho da Lâmina (para ambas as mãos) + Biomecânica com Hands
    if results_hands.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # Determinar se é mão direita ou esquerda
            if results_hands.multi_handedness:
                hand_label = results_hands.multi_handedness[hand_idx].classification[0].label
                is_right_hand = (hand_label == "Right")
            else:
                # Fallback: assumir que a primeira mão é direita e a segunda é esquerda
                is_right_hand = (hand_idx == 0)
            
            tip_lm = hand_landmarks.landmark[INDEX_FINGER_TIP_LM]
            tip_x, tip_y = int(tip_lm.x * frame_w), int(tip_lm.y * frame_h)

            blade_trail.append((tip_x, tip_y, time.time(), is_right_hand))

            if len(blade_trail) > 1:
                prev_x, prev_y, prev_is_right = blade_trail[-2][0], blade_trail[-2][1], blade_trail[-2][3]
                # Só verifica colisão se for o mesmo tipo de mão
                if prev_is_right == is_right_hand:
                    for fruit in fruits_on_screen:
                        check_collision_and_cut(fruit, tip_x, tip_y, prev_x, prev_y, is_right_hand)

            # Calcula velocidade da mão
            speed = calculate_hand_speed(hand_landmarks, hand_label, frame_w, frame_h, delta_time)
            
            # Acumula tempo em movimento
            side = 'right' if is_right_hand else 'left'
            calculate_moving_time(speed, side, delta_time)

    # Biomecânica com Pose (Ângulo de abertura do braço)
    if results_pose.pose_landmarks:
        calculate_arm_opening(results_pose.pose_landmarks, frame_w, frame_h, 'left')
        calculate_arm_opening(results_pose.pose_landmarks, frame_w, frame_h, 'right')

    # Limpa o rastro mantendo separado por mão
    current_time = time.time()
    blade_trail = [p for p in blade_trail if current_time - p[2] < 0.2]
    
    # Desenha o rastro para cada mão com cores diferentes
    right_trail = [p for p in blade_trail if p[3]]  # Mão direita
    left_trail = [p for p in blade_trail if not p[3]]  # Mão esquerda
    
    if len(right_trail) > 1:
        points = np.array([[p[0], p[1]] for p in right_trail], dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=False, color=(255, 100, 100), thickness=8, lineType=cv2.LINE_AA)
    
    if len(left_trail) > 1:
        points = np.array([[p[0], p[1]] for p in left_trail], dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=False, color=(100, 100, 255), thickness=8, lineType=cv2.LINE_AA)

    # Lógica e Desenho das Frutas
    if time.time() - last_spawn_time > SPAWN_INTERVAL:
        spawn_fruit(frame_w, frame_h)
        last_spawn_time = time.time()

    fruits_to_keep = []
    for fruit in fruits_on_screen:
        update_fruit_position(fruit)
        draw_fruit_3d(frame, fruit)
        if fruit['y'] < frame_h + fruit['prop']['radius'] * 2:
            fruits_to_keep.append(fruit)
    fruits_on_screen = fruits_to_keep

    # cria histórico de abertura
    if results_pose.pose_landmarks:
        left_angle = calculate_arm_opening(results_pose.pose_landmarks, frame_w, frame_h, 'left')
        right_angle = calculate_arm_opening(results_pose.pose_landmarks, frame_w, frame_h, 'right')

        if left_angle > 0:
            arm_opening_history['left'].append(left_angle)
        if right_angle > 0:
            arm_opening_history['right'].append(right_angle)


    # Calcula o tempo decorrido
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"Tempo: {minutes:02d}:{seconds:02d}"

    # Interface - Mostra pontuação separada e tempo
    cv2.putText(frame, f"Mao Direita: {SCORE_RIGHT}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(frame, f"Mao Direita: {SCORE_RIGHT}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 100, 255), 2)
    
    cv2.putText(frame, f"Mao Esquerda: {SCORE_LEFT}", (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(frame, f"Mao Esquerda: {SCORE_LEFT}", (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 100, 100), 2)
    
    cv2.putText(frame, time_str, (frame_w - 300, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(frame, time_str, (frame_w - 300, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

    # Mostra avaliações biomecânicas
    # Velocidade das mãos
    cv2.putText(frame, f"Vel. Dir: {hand_speeds['right']:.1f} px/s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, f"Vel. Dir: {hand_speeds['right']:.1f} px/s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 1)
    
    cv2.putText(frame, f"Vel. Esq: {hand_speeds['left']:.1f} px/s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, f"Vel. Esq: {hand_speeds['left']:.1f} px/s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 1)

    # Tempo em movimento
    cv2.putText(frame, f"Tempo. Mov. Dir: {moving_times['right']:.1f}s", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, f"Tempo. Mov. Dir: {moving_times['right']:.1f}s", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 1)
    
    cv2.putText(frame, f"Tempo. Mov. Esq: {moving_times['left']:.1f}s", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, f"Tempo. Mov. Esq: {moving_times['left']:.1f}s", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 1)

    # Abertura do braço
    cv2.putText(frame, f"Abert. Esq: {arm_openings['right']:.1f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, f"Abert. Esq: {arm_openings['right']:.1f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 1)
    
    cv2.putText(frame, f"Abert. Dir: {arm_openings['left']:.1f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, f"Abert. Dir: {arm_openings['left']:.1f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 1)

    # Adiciona a mensagem para sair
    cv2.putText(frame, "Pressione Q para sair", (frame_w - 400, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Fruit Ninja 3D - Duas Mãos", frame)

    # Fecha automaticamente após 2 minutos (120 segundos)
    if elapsed_time >= 120:
        print("⏰ Tempo limite de 2 minutos atingido. Encerrando jogo...")
        save_game_metrics_to_csv()
        break

    # Continua permitindo sair com a tecla Q (opcional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save_game_metrics_to_csv()
        break


cap.release()
cv2.destroyAllWindows()
