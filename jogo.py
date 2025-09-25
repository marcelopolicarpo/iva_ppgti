import cv2
import mediapipe as mp
import pygame
import sys

# ------------------ Configuração do MediaPipe ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Captura da webcam
cap = cv2.VideoCapture(0)

# ------------------ Configuração do Pygame ------------------
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Block Breaker Controlado pela Mão")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# Cores
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (200,0,0)
BLUE = (0,0,200)

# Barra
bar_width = 120
bar_height = 20
bar_y = HEIGHT - 40
bar_x = WIDTH // 2 - bar_width // 2

# Bola
ball_radius = 10
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_speed_x = 5
ball_speed_y = -5

# Blocos
rows = 5
cols = 8
block_width = WIDTH // cols
block_height = 30
blocks = []
for i in range(rows):
    for j in range(cols):
        blocks.append(pygame.Rect(j*block_width, i*block_height, block_width-5, block_height-5))

# ------------------ Função para detectar posição da mão ------------------
def get_hand_x():
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        # Pegando coordenada x do ponto médio da mão (punho)
        x = hand_landmarks.landmark[0].x  # valor entre 0 e 1
        return int(x * WIDTH)
    return None

# ------------------ Loop principal ------------------
running = True
while running:
    screen.fill(BLACK)
    
    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # ----------------- Atualiza posição da barra -----------------
    hand_x = get_hand_x()
    if hand_x is not None:
        bar_x = hand_x - bar_width // 2
    
    # Limita a barra dentro da tela
    bar_x = max(0, min(WIDTH-bar_width, bar_x))
    
    # ----------------- Atualiza posição da bola -----------------
    ball_x += ball_speed_x
    ball_y += ball_speed_y
    
    # Colisão com paredes
    if ball_x <= 0 or ball_x >= WIDTH:
        ball_speed_x *= -1
    if ball_y <= 0:
        ball_speed_y *= -1
    if ball_y >= HEIGHT:
        # Bola caiu, reinicia
        ball_x, ball_y = WIDTH//2, HEIGHT//2
        ball_speed_y *= -1
    
    # Colisão com barra
    bar_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
    ball_rect = pygame.Rect(ball_x-ball_radius, ball_y-ball_radius, ball_radius*2, ball_radius*2)
    if ball_rect.colliderect(bar_rect):
        ball_speed_y *= -1
    
    # Colisão com blocos
    for block in blocks[:]:
        if ball_rect.colliderect(block):
            ball_speed_y *= -1
            blocks.remove(block)
    
    # ----------------- Desenho -----------------
    pygame.draw.rect(screen, BLUE, bar_rect)
    pygame.draw.circle(screen, RED, (ball_x, ball_y), ball_radius)
    for block in blocks:
        pygame.draw.rect(screen, WHITE, block)
    
    # Atualiza tela
    pygame.display.flip()
    clock.tick(60)

# ------------------ Encerra ------------------
cap.release()
pygame.quit()
sys.exit()
