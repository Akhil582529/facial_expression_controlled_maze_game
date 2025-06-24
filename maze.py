import pygame
import cv2
import threading
import time
import numpy as np
from collections import deque
from keras.models import load_model

# === Load your trained model here ===
# model = load_model('your_model_path.h5')  # Uncomment and use actual model

# === Maze Setup ===
maze = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,1,0,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,1,0,1,1,0,1,0,1,0,1,1,0,1],
    [1,0,1,0,1,0,0,0,0,1,0,0,1,0,1],
    [1,0,1,0,1,1,1,1,0,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,1,0,1],
    [1,1,1,1,1,1,0,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,'G',1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]

# === Game Initialization ===
pygame.init()
WIDTH, HEIGHT = 600, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Emotion-Controlled Maze")
cell_size = WIDTH // len(maze[0])
player_pos = [1, 1]  # Starting position

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)

# === Emotion Detection Setup ===
current_emotion = "neutral"
emotion_queue = deque(maxlen=5)
emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']  # Adjust based on your model

def get_smoothed_emotion(new_emotion):
    emotion_queue.append(new_emotion)
    return max(set(emotion_queue), key=emotion_queue.count)

def detect_emotion():
    global current_emotion
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(cv2.resize(frame, (48, 48)), cv2.COLOR_BGR2GRAY)
        face = gray.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # === Your actual model prediction code ===
        # predictions = model.predict(face)
        # emotion_index = np.argmax(predictions)
        # confidence = np.max(predictions)
        # if confidence > 0.7:
        #     detected_emotion = emotion_labels[emotion_index]
        # else:
        #     detected_emotion = "neutral"

        # === Simulated prediction for testing ===
        detected_emotion = np.random.choice(emotion_labels)

        current_emotion = get_smoothed_emotion(detected_emotion)
        print("Detected (smoothed):", current_emotion)

        time.sleep(0.8)

# Start emotion detection in background thread
threading.Thread(target=detect_emotion, daemon=True).start()

# === Game Drawing Function ===
def draw_maze():
    win.fill(BLACK)
    for i, row in enumerate(maze):
        for j, val in enumerate(row):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            if val == 1:
                pygame.draw.rect(win, BLACK, rect)
            elif val == 0:
                pygame.draw.rect(win, WHITE, rect)
            elif val == 'G':
                pygame.draw.rect(win, GREEN, rect)
    pygame.draw.rect(win, BLUE, (player_pos[1]*cell_size, player_pos[0]*cell_size, cell_size, cell_size))
    pygame.display.update()

# === Movement Control ===
def move_player(emotion):
    x, y = player_pos
    if emotion == "surprise" and y+1 < len(maze[0]) and maze[x][y+1] != 1: #right
        player_pos[1] += 1
    elif emotion == "angry" and y-1 >= 0 and maze[x][y-1] != 1: #left
        player_pos[1] -= 1
    elif emotion == "sad" and x-1 >= 0 and maze[x-1][y] != 1: #up
        player_pos[0] -= 1
    elif emotion == "happy" and x+1 < len(maze) and maze[x+1][y] != 1: #down
        player_pos[0] += 1

# === Game Loop ===
clock = pygame.time.Clock()
running = True
last_move_time = time.time()

while running:
    draw_maze()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move every 1.2 seconds (cooldown)
    if time.time() - last_move_time > 1.2:
        move_player(current_emotion)
        last_move_time = time.time()

    # Check for win
    if maze[player_pos[0]][player_pos[1]] == 'G':
        print("🎉 You reached the goal!")
        running = False

    clock.tick(30)

pygame.quit()
