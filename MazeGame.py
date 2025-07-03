import cv2
import numpy as np
import pygame
import time
from collections import deque
from tensorflow.keras.models import load_model
import threading

class EmotionDetector:
    def __init__(self):
        self.active_emotion = "Neutral"  # This will always hold the most recent detected label
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = load_model('model/emotion_model.h5')
        
        # Camera setup with optimized settings
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.7)
        
        # Enhanced face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        
        # Emotion tracking
        self.current_emotion = "Neutral"
        self.emotion_buffer = deque(maxlen=9)
        self.running = True
        
        # Expression-specific settings
        self.thresholds = {'Angry': 0.65, 'Sad': 0.6, 'Happy': 0.55, 'Surprise': 0.55, 'Neutral': 0.4}
        self.colors = {
            'Angry': (0, 0, 255),    # Red
            'Sad': (255, 0, 255),    # Purple
            'Happy': (0, 255, 0),    # Green
            'Surprise': (0, 255, 255), # Yellow
            'Neutral': (255, 255, 255) # White
        }

    def get_smoothed_prediction(self, label, confidence):
        """Weighted smoothing favoring difficult expressions"""
        weights = {'Angry': 1.3, 'Sad': 1.2, 'Happy': 1.0, 'Surprise': 1.0, 'Neutral': 0.8}
        self.emotion_buffer.append((label, confidence * weights.get(label, 1.0)))
        
        emotion_weights = {}
        for emo, conf in self.emotion_buffer:
            emotion_weights[emo] = emotion_weights.get(emo, 0) + conf
        
        return max(emotion_weights.items(), key=lambda x: x[1])[0]

    def preprocess_face(self, face_roi):
        """Enhanced preprocessing for better emotion detection"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_roi = clahe.apply(face_roi)
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        roi = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi.astype('float32') / 255.0
        return np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)

    def detect_emotions(self):
        """Main detection loop with visual feedback"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=7, minSize=(120, 120))
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                face_roi = gray[y:y+h, x:x+w]
                
                try:
                    roi = self.preprocess_face(face_roi)
                    predictions = self.model.predict(roi, verbose=0)[0]
                    confidence = np.max(predictions)
                    label = self.emotions[np.argmax(predictions)]
                    
                    self.active_emotion = label  # Always update
                    if confidence > self.thresholds.get(label, 0.6):
                        self.current_emotion = self.get_smoothed_prediction(label, confidence)
                    
                    # Visual feedback
                    color = self.colors.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Expression-specific guidance
                    if label == 'Sad':
                        cv2.putText(frame, "Droop mouth corners", 
                                  (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    elif label == 'Angry':
                        cv2.putText(frame, "Furrow brows", 
                                  (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                except Exception as e:
                    print(f"Processing error: {e}")

            # General instructions
            cv2.putText(frame, "Make exaggerated expressions", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, "Press Q to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        
        self.cap.release()
        cv2.destroyAllWindows()

class MazeGame:
    def __init__(self):
        pygame.init()
        self.maze = [
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
        
        self.width, self.height = 600, 600
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Emotion-Controlled Maze")
        self.cell_size = self.width // len(self.maze[0])
        self.player_pos = [1, 1]
        self.last_move_time = time.time()
        self.move_cooldown = 1.0
        
        self.COLORS = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'wall': (100, 100, 100)
        }
        
        self.detector = EmotionDetector()
        self.detector_thread = threading.Thread(target=self.detector.detect_emotions, daemon=True)
        self.detector_thread.start()

    def draw_maze(self):
        self.win.fill(self.COLORS['white'])
        for i, row in enumerate(self.maze):
            for j, val in enumerate(row):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, 
                                 self.cell_size, self.cell_size)
                if val == 1:
                    pygame.draw.rect(self.win, self.COLORS['wall'], rect)
                elif val == 'G':
                    pygame.draw.rect(self.win, self.COLORS['green'], rect)
        
        pygame.draw.rect(self.win, self.COLORS['blue'], (
            self.player_pos[1]*self.cell_size + 2, 
            self.player_pos[0]*self.cell_size + 2, 
            self.cell_size - 4, self.cell_size - 4))
        
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Emotion: {self.detector.current_emotion}", True, self.COLORS['red'])
        self.win.blit(text, (10, 10))
        
        instruction_font = pygame.font.SysFont(None, 24)
        instructions = [
            "Controls:",
            "Happy = Down | Sad = Up",
            "Angry = Left | Surprise = Right"
        ]
        for i, line in enumerate(instructions):
            text = instruction_font.render(line, True, self.COLORS['black'])
            self.win.blit(text, (10, self.height - 80 + i*25))
        
        pygame.display.update()

    def move_player(self):
        x, y = self.player_pos
        new_x, new_y = x, y
        
        emotion = self.detector.active_emotion
        
        if emotion == "Surprise": new_y = y + 1
        if emotion == "Angry": new_y = y - 1
        if emotion == "Sad": new_x = x - 1
        if emotion == "Happy": new_x = x + 1
        
        if 0 <= new_x < len(self.maze) and 0 <= new_y < len(self.maze[0]):
            if self.maze[new_x][new_y] != 1:
                self.player_pos = [new_x, new_y]
                return True
        return False

    def check_win(self):
        return self.maze[self.player_pos[0]][self.player_pos[1]] == 'G'

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            current_time = time.time()
            if current_time - self.last_move_time > self.move_cooldown:
                if self.move_player():
                    self.last_move_time = current_time
            
            if self.check_win():
                font = pygame.font.SysFont(None, 72)
                text = font.render("YOU WIN!", True, self.COLORS['red'])
                self.win.blit(text, (self.width//2 - 100, self.height//2 - 36))
                pygame.display.update()
                # time.sleep(3)
                running = False
            
            self.draw_maze()
            clock.tick(15)
        
        self.detector.running = False
        self.detector_thread.join()
        pygame.quit()

if __name__ == "__main__":
    game = MazeGame()
    game.run()
