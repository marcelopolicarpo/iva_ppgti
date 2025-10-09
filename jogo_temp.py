import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
import pygame
import csv
from collections import deque

# =============================
#  Áudio (som de corte)
# =============================
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    knife_sound = pygame.mixer.Sound(r"./knife-cut.mp3")
    knife_sound.set_volume(0.6)
    AUDIO_OK = True
    print("✅ Efeito sonoro carregado com sucesso!")
except Exception as e:
    print(f"⚠️ Sem áudio disponível: {e}")
    knife_sound = None
    AUDIO_OK = False

# =============================
#  MediaPipe Hands + Pose
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # (fix) antes estava 'max_num_mands'
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =============================
#  Captura de Vídeo
# =============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Landmarks úteis
INDEX_FINGER_TIP_LM = mp_hands.HandLandmark.INDEX_FINGER_TIP
WRIST_LM = mp_hands.HandLandmark.WRIST

# =============================
#  Propriedades das Frutas (3D)
# =============================
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
            'rind': (210, 230, 210)
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

# =============================
#  Estado do Jogo e Métricas
# =============================
fruits_on_screen = []
last_spawn_time = time.time()
SPAWN_INTERVAL = 1.2
MAX_FRUITS_ON_SCREEN = 3
SCORE_RIGHT = 0
SCORE_LEFT = 0
start_time = time.time()
blade_trail = []  # (x, y, t, is_right)

# Métricas biomecânicas
prev_time = time.time()
prev_wrist_positions = {'left': None, 'right': None}
velocity_hist = {'left': deque(maxlen=5), 'right': deque(maxlen=5)}  # suavização (média móvel)
hand_speeds = {'left': 0.0, 'right': 0.0}  # px/s (média dos últimos frames)
up_start = {'left': None, 'right': None}
up_time = {'left': 0.0, 'right': 0.0}
arm_opening = {'left': 0.0, 'right': 0.0}  # ângulo braço vs tronco (graus)
MOVING_SPEED_THRESHOLD = 50  # px/s (para considerar "movendo")

# =============================
#  CSV em tempo real
# =============================
CSV_PATH = 'metricas.csv'
try:
    csv_file = open(CSV_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['tempo', 'mao', 'angulo', 'velocidade_px_s', 'tempo_up_s'])
    csv_file.flush()
except Exception as e:
    print(f"⚠️ Não foi possível abrir {CSV_PATH} para escrita: {e}")
    csv_file = None
    csv_writer = None

# =============================
#  Funções do Jogo
# =============================

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
            if knife_sound is not None:
                try:
                    knife_sound.play()
                except Exception:
                    pass
        fruit['status'] = 'cut'
        fruit['cut_angle'] = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.pi / 2
        fruit['vy'] -= 8
        cut_strength = 6
        fruit['cut_impulse_x'] = math.cos(fruit['cut_angle']) * cut_strength
        fruit['cut_impulse_y'] = math.sin(fruit['cut_angle']) * cut_strength
        return True
    return False

# =============================
#  Desenho 3D
# =============================

def draw_3d_sphere(frame, center, radius, colors):
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
    cv2.circle(frame, (x1, y1), half_r, inner_color, -1)
    cv2.circle(frame, (x2, y2), half_r, inner_color, -1)

    if fruit['prop']['id'] == 'watermelon':
        rind_thickness = int(half_r * 0.1)
        cv2.circle(frame, (x1, y1), half_r, colors['rind'], rind_thickness)
        cv2.circle(frame, (x2, y2), half_r, colors['rind'], rind_thickness)
        for i in range(3):
            angle = random.uniform(0, math.pi * 2)
            dist = random.uniform(0, half_r * 0.7)
            cv2.circle(frame, (int(x1 + dist * math.cos(angle)), int(y1 + dist * math.sin(angle))), 2, colors['seed'], -1)
            cv2.circle(frame, (int(x2 + dist * math.cos(angle)), int(y2 + dist * math.sin(angle))), 2, colors['seed'], -1)

    elif fruit['prop']['id'] == 'orange':
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
        draw_3d_sphere(frame, (x, y), r, colors)
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
            cv2.circle(frame, (x, y - r + 5), 3, colors['dark'], -1)
            cv2.circle(frame, (x - r // 3, y - r // 3), r // 5, colors['highlight'], -1, lineType=cv2.LINE_AA)
        elif fruit['prop']['id'] == 'bomb':
            fuse_end = (x + r // 2, y - r - r // 4)
            cv2.line(frame, (x, y - r), fuse_end, colors['fuse_color'], 3)
            if time.time() % 0.2 < 0.1:
                cv2.circle(frame, fuse_end, 7, colors['spark_color'], -1)
            cv2.circle(frame, (x - r // 3, y - r // 3), r // 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    else:
        draw_3d_cut_fruit(frame, fruit)

# =============================
#  Biomecânica: funções
# =============================

def update_hand_up_time(hand_label, landmarks):
    """Soma tempo com a mão levantada (ponta do indicador acima do punho)."""
    side = 'right' if hand_label == 'Right' else 'left'
    idx = landmarks[INDEX_FINGER_TIP_LM]
    wr = landmarks[WRIST_LM]
    t = time.time()
    up = idx.y < wr.y
    if up:
        if up_start[side] is None:
            up_start[side] = t
    else:
        if up_start[side] is not None:
            up_time[side] += t - up_start[side]
            up_start[side] = None


def update_hand_speed(hand_label, landmarks, frame_w, frame_h, dt):
    """Atualiza velocidade média (px/s) do punho com média móvel (5 amostras)."""
    side = 'right' if hand_label == 'Right' else 'left'
    w = landmarks[WRIST_LM]
    cur = np.array([w.x * frame_w, w.y * frame_h], dtype=np.float32)
    prev = prev_wrist_positions[side]
    if prev is not None and dt > 0:
        vel = float(np.linalg.norm(cur - prev) / dt)
        velocity_hist[side].append(vel)
        hand_speeds[side] = float(np.mean(velocity_hist[side]))
    prev_wrist_positions[side] = cur


def update_arm_opening(pose_landmarks, frame_w, frame_h):
    """Ângulo entre braço (ombro→cotovelo) e tronco (ombro→quadril), por lado."""
    if pose_landmarks is None:
        return
    L = mp_pose.PoseLandmark
    pts = pose_landmarks.landmark
    for side, s_sh, s_el, s_hp in (
        ('left', L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_HIP),
        ('right', L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_HIP),
    ):
        if pts[s_sh].visibility < 0.4 or pts[s_el].visibility < 0.4 or pts[s_hp].visibility < 0.4:
            arm_opening[side] = 0.0
            continue
        sh = np.array([pts[s_sh].x * frame_w, pts[s_sh].y * frame_h], dtype=np.float32)
        el = np.array([pts[s_el].x * frame_w, pts[s_el].y * frame_h], dtype=np.float32)
        hp = np.array([pts[s_hp].x * frame_w, pts[s_hp].y * frame_h], dtype=np.float32)
        v_arm = el - sh
        v_body = hp - sh
        n1, n2 = np.linalg.norm(v_arm), np.linalg.norm(v_body)
        if n1 == 0 or n2 == 0:
            arm_opening[side] = 0.0
        else:
            cosang = float(np.clip(np.dot(v_arm, v_body) / (n1 * n2), -1.0, 1.0))
            arm_opening[side] = float(math.degrees(math.acos(cosang)))

# =============================
#  Indicadores visuais (barras)
# =============================

def draw_bar_vertical(frame, x, y, w, h, value, vmin, vmax, color):
    value = max(vmin, min(vmax, value))
    frac = 0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    filled = int(h * frac)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), 2)
    cv2.rectangle(frame, (x + 2, y + h - filled), (x + w - 2, y + h - 2), color, -1)


def draw_bar_horizontal(frame, x, y, w, h, value, vmin, vmax, color):
    value = max(vmin, min(vmax, value))
    frac = 0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    filled = int(w * frac)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), 2)
    cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + filled, y + h - 2), color, -1)

# =============================
#  Loop Principal
# =============================
cv2.namedWindow("Fruit Ninja 3D - Duas Mãos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fruit Ninja 3D - Duas Mãos", 1280, 720)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_h = hands.process(rgb)
        res_p = pose.process(rgb)

        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Hands
        if res_h.multi_hand_landmarks:
            for i, hand_lms in enumerate(res_h.multi_hand_landmarks):
                if res_h.multi_handedness and i < len(res_h.multi_handedness):
                    hand_label = res_h.multi_handedness[i].classification[0].label  # 'Right'/'Left'
                else:
                    hand_label = 'Right' if i == 0 else 'Left'

                # tempo "up" (indicador acima do punho)
                update_hand_up_time(hand_label, hand_lms.landmark)
                # velocidade média (px/s)
                update_hand_speed(hand_label, hand_lms.landmark, frame_w, frame_h, dt)

                # trilha e colisões
                tip = hand_lms.landmark[INDEX_FINGER_TIP_LM]
                tx, ty = int(tip.x * frame_w), int(tip.y * frame_h)
                is_right = (hand_label == 'Right')
                blade_trail.append((tx, ty, now, is_right))
                if len(blade_trail) > 1 and blade_trail[-2][3] == is_right:
                    px, py = blade_trail[-2][0], blade_trail[-2][1]
                    for fruit in fruits_on_screen:
                        check_collision_and_cut(fruit, tx, ty, px, py, is_right)

        # Pose → ângulo do braço em relação ao tronco
        if res_p.pose_landmarks:
            update_arm_opening(res_p.pose_landmarks, frame_w, frame_h)

        # limpeza da trilha
        blade_trail = [p for p in blade_trail if now - p[2] < 0.2]
        right_trail = [p for p in blade_trail if p[3]]
        left_trail  = [p for p in blade_trail if not p[3]]
        if len(right_trail) > 1:
            pts = np.array([[p[0], p[1]] for p in right_trail], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(255, 100, 100), thickness=8, lineType=cv2.LINE_AA)
        if len(left_trail) > 1:
            pts = np.array([[p[0], p[1]] for p in left_trail], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(100, 100, 255), thickness=8, lineType=cv2.LINE_AA)

        # Frutas
        if time.time() - last_spawn_time > SPAWN_INTERVAL:
            spawn_fruit(frame_w, frame_h)
            last_spawn_time = time.time()
        kept = []
        for fruit in fruits_on_screen:
            update_fruit_position(fruit)
            draw_fruit_3d(frame, fruit)
            if fruit['y'] < frame_h + fruit['prop']['radius'] * 2:
                kept.append(fruit)
        fruits_on_screen = kept

        # HUD texto
        elapsed = now - start_time
        m, s = int(elapsed // 60), int(elapsed % 60)
        time_str = f"Tempo: {m:02d}:{s:02d}"

        cv2.putText(frame, f"Mao Direita: {SCORE_RIGHT}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, f"Mao Direita: {SCORE_RIGHT}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 100, 255), 2)

        cv2.putText(frame, f"Mao Esquerda: {SCORE_LEFT}", (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, f"Mao Esquerda: {SCORE_LEFT}", (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 100, 100), 2)

        cv2.putText(frame, time_str, (frame_w - 300, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, time_str, (frame_w - 300, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

        cv2.putText(frame, f"Dir: vel {hand_speeds['right']:.0f}px/s up {up_time['right']:.1f}s ang {arm_opening['right']:.1f}°", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(frame, f"Dir: vel {hand_speeds['right']:.0f}px/s up {up_time['right']:.1f}s ang {arm_opening['right']:.1f}°", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 1)

        cv2.putText(frame, f"Esq: vel {hand_speeds['left']:.0f}px/s up {up_time['left']:.1f}s ang {arm_opening['left']:.1f}°", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(frame, f"Esq: vel {hand_speeds['left']:.0f}px/s up {up_time['left']:.1f}s ang {arm_opening['left']:.1f}°", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 1)

        cv2.putText(frame, "Pressione Q para sair", (frame_w - 400, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Indicadores visuais (barras)
        ANG_MAX = 180.0
        VEL_MAX = 800.0  # ajuste se necessário
        # barras verticais de ângulo nas laterais
        draw_bar_vertical(frame, 20, 80, 20, 180, arm_opening['left'], 0, ANG_MAX, (255, 150, 150))
        draw_bar_vertical(frame, frame_w - 40, 80, 20, 180, arm_opening['right'], 0, ANG_MAX, (150, 200, 255))
        # barras horizontais de velocidade embaixo
        draw_bar_horizontal(frame, 20, frame_h - 60, 250, 18, hand_speeds['left'], 0, VEL_MAX, (255, 150, 150))
        draw_bar_horizontal(frame, frame_w - 270, frame_h - 60, 250, 18, hand_speeds['right'], 0, VEL_MAX, (150, 200, 255))

        # CSV (duas linhas por frame: Right/Left)
        if csv_writer is not None:
            csv_writer.writerow([f"{elapsed:.2f}", 'Right', f"{arm_opening['right']:.2f}", f"{hand_speeds['right']:.2f}", f"{up_time['right']:.2f}"])
            csv_writer.writerow([f"{elapsed:.2f}", 'Left',  f"{arm_opening['left']:.2f}",  f"{hand_speeds['left']:.2f}",  f"{up_time['left']:.2f}"])
            csv_file.flush()

        cv2.imshow("Fruit Ninja 3D - Duas Mãos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # fecha períodos abertos de mão levantada
    t = time.time()
    for side in ('left', 'right'):
        if up_start[side] is not None:
            up_time[side] += t - up_start[side]
            up_start[side] = None

    try:
        cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    try:
        if csv_file is not None:
            csv_file.close()
    except:
        pass
    try:
        if AUDIO_OK:
            pygame.mixer.quit()
    except:
        pass

    print(f"📊 Métricas salvas continuamente em {CSV_PATH}")
