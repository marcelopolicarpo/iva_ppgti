import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
import pygame

# =============================
#  Áudio (som de corte)
# =============================
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    AUDIO_OK = True
except Exception as e:
    print(f"⚠️ Sem áudio disponível: {e}")
    AUDIO_OK = False

knife_sound = None
if AUDIO_OK:
    try:
        knife_sound = pygame.mixer.Sound(r"./knife-cut.mp3")
        knife_sound.set_volume(0.6)
        print("✅ Efeito sonoro carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar efeito sonoro: {e}")
        knife_sound = None

# =============================
#  MediaPipe Hands
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# =============================
#  Captura de Vídeo
# =============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# IDs dos pontos de referência da mão
INDEX_FINGER_TIP_LM = mp_hands.HandLandmark.INDEX_FINGER_TIP

# =============================
#  Propriedades das Frutas
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
#  Estado Global do Jogo
# =============================
fruits_on_screen = []
last_spawn_time = time.time()
SPAWN_INTERVAL = 1.2
MAX_FRUITS_ON_SCREEN = 3
SCORE_RIGHT = 0
SCORE_LEFT = 0
start_time = time.time()
blade_trail = []  # (x, y, t, is_right)
last_cut_speed = 0.0

# Métricas biomecânicas exibidas no HUD
hand_lift_start_time = {'Right': None, 'Left': None}
hand_lift_duration = {'Right': 0.0, 'Left': 0.0}
max_arm_angle = {'Right': 0.0, 'Left': 0.0}  # ângulo de abertura da mão

# =============================
#  Funções Biomecânicas (Hands)
# =============================
def tempo_mao_levantada(hand_label, landmarks):
    global hand_lift_start_time, hand_lift_duration
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    up = index_tip.y < wrist.y
    t = time.time()
    if up:
        if hand_lift_start_time[hand_label] is None:
            hand_lift_start_time[hand_label] = t
    else:
        if hand_lift_start_time[hand_label] is not None:
            hand_lift_duration[hand_label] += t - hand_lift_start_time[hand_label]
            hand_lift_start_time[hand_label] = None


def angulo_abertura_mao(hand_label, landmarks):
    global max_arm_angle
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x,
                      landmarks[mp_hands.HandLandmark.WRIST].y])
    idx_mcp = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                        landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])
    min_mcp = np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP].x,
                        landmarks[mp_hands.HandLandmark.PINKY_MCP].y])
    v1 = idx_mcp - wrist
    v2 = min_mcp - wrist
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        angle_deg = 0.0
    else:
        cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cosang))
    if angle_deg > max_arm_angle[hand_label]:
        max_arm_angle[hand_label] = angle_deg

# =============================
#  Lógica do Jogo
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
    seed = random.randint(0, 2**31 - 1)  # seed por fruta p/ estabilidade visual
    fruits_on_screen.append({
        'prop': prop, 'x': x, 'y': y, 'vx': speed_x, 'vy': speed_y,
        'gravity': gravity, 'status': 'active', 'cut_angle': 0.0,
        'cut_impulse_x': 0.0, 'cut_impulse_y': 0.0,
        'seed': seed
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
    global SCORE_RIGHT, SCORE_LEFT, last_cut_speed
    if fruit['status'] != 'active':
        return False

    fruit_center = np.array([fruit['x'], fruit['y']], dtype=np.float32)
    p1 = np.array([prev_tip_x, prev_tip_y], dtype=np.float32)
    p2 = np.array([current_tip_x, current_tip_y], dtype=np.float32)

    line_vec = p2 - p1
    fruit_vec = fruit_center - p1
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq == 0.0:
        distance = float(np.linalg.norm(fruit_center - p1))
    else:
        t = max(0.0, min(1.0, float(np.dot(fruit_vec, line_vec) / line_len_sq)))
        closest_point_on_line = p1 + t * line_vec
        distance = float(np.linalg.norm(fruit_center - closest_point_on_line))

    speed = float(np.linalg.norm(line_vec))  # ~pixels por frame
    min_speed = 18.0
    eff_radius = fruit['prop']['radius'] * 0.85

    if speed < min_speed:
        return False

    if distance < eff_radius:
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
            last_cut_speed = speed
            print(f"✔️ Corte! Pontuação - Direita: {SCORE_RIGHT}, Esquerda: {SCORE_LEFT}")
            if knife_sound is not None:
                try:
                    knife_sound.play(fade_ms=10)
                except Exception:
                    pass

        fruit['status'] = 'cut'
        fruit['cut_angle'] = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])) + math.pi / 2
        fruit['vy'] -= 8
        cut_strength = 6
        fruit['cut_impulse_x'] = math.cos(fruit['cut_angle']) * cut_strength
        fruit['cut_impulse_y'] = math.sin(fruit['cut_angle']) * cut_strength
        return True
    return False

# =============================
#  Desenho 3D e Efeitos
# =============================
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

    cv2.circle(frame, (x1, y1), half_r, inner_color, -1)
    cv2.circle(frame, (x2, y2), half_r, inner_color, -1)

    rng = random.Random(fruit.get('seed', 0))

    if fruit['prop']['id'] == 'watermelon':
        rind_thickness = int(half_r * 0.1)
        cv2.circle(frame, (x1, y1), half_r, colors['rind'], rind_thickness)
        cv2.circle(frame, (x2, y2), half_r, colors['rind'], rind_thickness)
        for i in range(3):
            angle = rng.uniform(0, math.pi * 2)
            dist = rng.uniform(0, half_r * 0.7)
            cv2.circle(frame, (int(x1 + dist * math.cos(angle)), int(y1 + dist * math.sin(angle))), 2, colors['seed'], -1)
            cv2.circle(frame, (int(x2 + dist * math.cos(angle)), int(y2 + dist * math.sin(angle))), 2, colors['seed'], -1)

    elif fruit['prop']['id'] == 'orange':
        for i in range(8):
            ang = i * (math.pi / 4)
            cv2.line(frame, (x1, y1), (int(x1 + math.cos(ang) * half_r), int(y1 + math.sin(ang) * half_r)), colors['highlight'], 1)
            cv2.line(frame, (x2, y2), (int(x2 + math.cos(ang) * half_r), int(y2 + math.sin(ang) * half_r)), colors['highlight'], 1)

    angle_deg = math.degrees(angle_rad)
    cv2.ellipse(frame, (x1, y1), (half_r, half_r), angle_deg, 90, 270, colors['base'], -1)
    cv2.ellipse(frame, (x2, y2), (half_r, half_r), angle_deg, -90, 90, colors['base'], -1)


def draw_fruit_3d(frame, fruit):
    x, y, r = int(fruit['x']), int(fruit['y']), fruit['prop']['radius']
    colors = fruit['prop']['colors']
    rng = random.Random(fruit.get('seed', 0))

    cv2.circle(frame, (x, y + r//2), int(r * 0.9), (30, 30, 30), -1)

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
                jitter = rng.uniform(-5, 5)
                for j in range(20):
                    a = math.radians(angle + jitter)
                    rad = r * (1 - (j / 20) ** 2 * 0.5 + rng.uniform(-0.05, 0.05))
                    px = int(x + rad * math.sin(a) * math.cos(math.radians(j * 10)))
                    py = int(y - rad * math.cos(a))
                    if j > 0:
                        cv2.line(frame, (pts[-1][0], pts[-1][1]), (px, py), colors['stripe_color'], int(r * 0.08))
                    pts.append([px, py])

        elif fruit['prop']['id'] == 'orange':
            for _ in range(30):
                angle, dist = rng.uniform(0, 2 * math.pi), rng.uniform(0, r * 0.9)
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

    elif fruit['status'] == 'cut':
        draw_3d_cut_fruit(frame, fruit)


def draw_trail(frame, trail, color=(255, 255, 255), base_thick=10, ttl=0.2):
    for i in range(1, len(trail)):
        x1, y1, t1, _ = trail[i-1]
        x2, y2, t2, _ = trail[i]
        age = time.time() - t2
        alpha = max(0.0, 1.0 - age/ttl)
        thick = max(2, int(base_thick * 1.6 * alpha))
        cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), thick, lineType=cv2.LINE_AA)
    for i in range(1, len(trail)):
        x1, y1, t1, _ = trail[i-1]
        x2, y2, t2, _ = trail[i]
        age = time.time() - t2
        alpha = max(0.0, 1.0 - age/ttl)
        thick = max(2, int(base_thick * alpha))
        cv2.line(frame, (x1, y1), (x2, y2), color, thick, lineType=cv2.LINE_AA)

# =============================
#  Loop Principal
# =============================
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Rastreamento e colisões
        if results.multi_hand_landmarks:
            n_hands = len(results.multi_hand_landmarks)
            for hand_idx in range(n_hands):
                hand_landmarks = results.multi_hand_landmarks[hand_idx]

                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    hand_label = results.multi_handedness[hand_idx].classification[0].label
                    is_right_hand = (hand_label == "Right")
                else:
                    hand_label = "Right" if hand_idx == 0 else "Left"
                    is_right_hand = (hand_label == "Right")

                # métricas biomecânicas do HUD
                tempo_mao_levantada(hand_label, hand_landmarks.landmark)
                angulo_abertura_mao(hand_label, hand_landmarks.landmark)

                tip_lm = hand_landmarks.landmark[INDEX_FINGER_TIP_LM]
                tip_x, tip_y = int(tip_lm.x * frame_w), int(tip_lm.y * frame_h)

                blade_trail.append((tip_x, tip_y, time.time(), is_right_hand))

                if len(blade_trail) > 1 and blade_trail[-2][3] == is_right_hand:
                    prev_x, prev_y = blade_trail[-2][0], blade_trail[-2][1]
                    for fruit in fruits_on_screen:
                        check_collision_and_cut(fruit, tip_x, tip_y, prev_x, prev_y, is_right_hand)

        current_time = time.time()
        blade_trail = [p for p in blade_trail if current_time - p[2] < 0.2]

        right_trail = [p for p in blade_trail if p[3]]
        left_trail = [p for p in blade_trail if not p[3]]
        if len(right_trail) > 1:
            draw_trail(frame, right_trail, color=(255, 100, 100), base_thick=10, ttl=0.2)
        if len(left_trail) > 1:
            draw_trail(frame, left_trail, color=(100, 100, 255), base_thick=10, ttl=0.2)

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

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"Tempo: {minutes:02d}:{seconds:02d}"

        # HUD (pontuação, tempo, biomecânica e velocidade do golpe)
        cv2.putText(frame, f"Mao Direita: {SCORE_RIGHT}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, f"Mao Direita: {SCORE_RIGHT}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 100, 255), 2)

        cv2.putText(frame, f"Mao Esquerda: {SCORE_LEFT}", (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, f"Mao Esquerda: {SCORE_LEFT}", (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 100, 100), 2)

        cv2.putText(frame, time_str, (frame_w - 300, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(frame, time_str, (frame_w - 300, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

        cv2.putText(frame, f"Vel. golpe: {last_cut_speed:4.0f} px/f", (10, 180), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(frame, f"Vel. golpe: {last_cut_speed:4.0f} px/f", (10, 180), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (220, 255, 220), 2)

        cv2.putText(frame, f"Dir: up {hand_lift_duration['Right']:.1f}s maxAng {max_arm_angle['Right']:.1f}", (10, 210), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 5)
        cv2.putText(frame, f"Dir: up {hand_lift_duration['Right']:.1f}s maxAng {max_arm_angle['Right']:.1f}", (10, 210), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 255, 200), 2)

        cv2.putText(frame, f"Esq: up {hand_lift_duration['Left']:.1f}s maxAng {max_arm_angle['Left']:.1f}", (10, 235), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 5)
        cv2.putText(frame, f"Esq: up {hand_lift_duration['Left']:.1f}s maxAng {max_arm_angle['Left']:.1f}", (10, 235), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 200, 255), 2)

        cv2.putText(frame, "Pressione Q para sair", (frame_w - 400, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Fruit Ninja 3D - Duas Maos (Limpo)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # fechar períodos abertos de mão levantada
    t = time.time()
    for h in ['Right', 'Left']:
        if hand_lift_start_time[h] is not None:
            hand_lift_duration[h] += t - hand_lift_start_time[h]
            hand_lift_start_time[h] = None

    try:
        cap.release()
    except:
        pass
    try:
        hands.close()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    try:
        if AUDIO_OK:
            pygame.mixer.quit()
    except:
        pass
