import pygame
import cv2
import numpy as np
from collections import deque
import time
import threading
from tensorflow.keras.models import load_model

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
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WALL_COLOR = (100, 100, 100)

# === Enhanced Emotion Detection Setup ===
current_emotion = "Neutral"
emotion_buffer = deque(maxlen=7)  # Increased buffer size
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion model
try:
    model = load_model('model/emotion_model.h5')
    print("Emotion model loaded successfully")
except Exception as e:
    print(f"Error loading emotion model: {e}")
    exit()

def get_smoothed_prediction(label, confidence):
    # Weighted smoothing - higher confidence predictions have more impact
    emotion_buffer.append((label, confidence))
    
    # Get the most frequent emotion, weighted by confidence
    emotion_counts = {}
    for emo, conf in emotion_buffer:
        emotion_counts[emo] = emotion_counts.get(emo, 0) + conf
    
    return max(emotion_counts.items(), key=lambda x: x[1])[0]

def get_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])

def detect_emotion():
    global current_emotion
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Adjust camera settings for better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.6)

    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to grayscale and equalize histogram
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)
        
        # Detect faces with more sensitive parameters
        faces = face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        label = current_emotion  # default to current
        confidence = 0
        largest_face = get_largest_face(faces)

        if largest_face is not None:
            (x, y, w, h) = largest_face
            roi_gray = grayscale[y:y+h, x:x+w]
            try:
                roi = cv2.resize(roi_gray, (48, 48))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)

                prediction = model.predict(roi, verbose=0)[0]
                confidence = np.max(prediction)
                label_idx = np.argmax(prediction)
                label = emotions[label_idx]

                # Only update if confidence is reasonable
                if confidence > 0.4:  # Lowered threshold
                    current_emotion = get_smoothed_prediction(label, confidence)
                else:
                    # If low confidence, only update if not Neutral
                    if label != "Neutral":
                        current_emotion = get_smoothed_prediction(label, confidence)

                # Draw visualization
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'{current_emotion} ({confidence:.2f})', (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Draw face landmarks for better alignment check
                cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 0, 255), -1)
                
            except Exception as e:
                print("Face processing error:", e)

        # Show FPS and detection info
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Press Q to close', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Emotion Detection - Make Exaggerated Expressions', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start emotion detection thread
emotion_thread = threading.Thread(target=detect_emotion, daemon=True)
emotion_thread.start()

# === Game Drawing ===
def draw_maze():
    win.fill(WHITE)
    for i, row in enumerate(maze):
        for j, val in enumerate(row):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            if val == 1:
                pygame.draw.rect(win, WALL_COLOR, rect)
            elif val == 'G':
                pygame.draw.rect(win, GREEN, rect)
    
    pygame.draw.rect(win, BLUE, (player_pos[1]*cell_size + 2, player_pos[0]*cell_size + 2, 
                    cell_size - 4, cell_size - 4))
    
    # Enhanced emotion display
    font = pygame.font.SysFont(None, 36)
    text = font.render(f"Emotion: {current_emotion}", True, RED)
    win.blit(text, (10, 10))
    
    # Add instruction text
    instruction_font = pygame.font.SysFont(None, 24)
    instructions = [
        "Make exaggerated facial expressions:",
        "Happy = Down | Sad = Up | Angry = Left",
        "Surprise = Right | Fear = Up-Right | Disgust = Up-Left"
    ]
    for i, line in enumerate(instructions):
        text = instruction_font.render(line, True, BLACK)
        win.blit(text, (10, HEIGHT - 80 + i*25))
    
    pygame.display.update()

# === Movement Control ===
def move_player(emotion):
    x, y = player_pos
    new_x, new_y = x, y
    
    if emotion == "Surprise":  # Right
        new_y = y + 1
    elif emotion == "Angry":   # Left
        new_y = y - 1
    elif emotion == "Sad":     # Up
        new_x = x - 1
    elif emotion == "Happy":   # Down
        new_x = x + 1
    elif emotion == "Disgust": # Up-Left
        new_x, new_y = x - 1, y - 1
    elif emotion == "Fear":    # Up-Right
        new_x, new_y = x - 1, y + 1
    
    # Check bounds and walls
    if 0 <= new_x < len(maze) and 0 <= new_y < len(maze[0]):
        if maze[new_x][new_y] != 1:
            player_pos[0], player_pos[1] = new_x, new_y
            return True
    return False

# === Main Game Loop ===
clock = pygame.time.Clock()
running = True
last_move_time = time.time()
move_cooldown = 1.2  # seconds

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Movement with cooldown
    current_time = time.time()
    if current_time - last_move_time > move_cooldown:
        if move_player(current_emotion):
            last_move_time = current_time
    
    # Win condition
    if maze[player_pos[0]][player_pos[1]] == 'G':
        font = pygame.font.SysFont(None, 72)
        text = font.render("YOU WIN!", True, RED)
        win.blit(text, (WIDTH//2 - 100, HEIGHT//2 - 36))
        pygame.display.update()
        time.sleep(3)
        running = False
    
    draw_maze()
    clock.tick(30)

pygame.quit()
