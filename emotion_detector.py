import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

class EmotionDetector:
    def __init__(self):
        # Model configuration
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = load_model('model/emotion_model.h5')
        
        # Video capture setup
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        
        # Face detection
        self.face_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Processing buffers
        self.emotion_buffer = deque(maxlen=7)  # Odd number for tie-breaking
        self.last_prediction = 'Neutral'
        self.last_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance metrics
        self.avg_processing_time = 0
        self.frames_processed = 0

    def get_smoothed_prediction(self, label, confidence):
        """Weighted smoothing based on confidence scores"""
        self.emotion_buffer.append((label, confidence))
        
        # Calculate weighted frequencies
        emotion_weights = {}
        for emo, conf in self.emotion_buffer:
            emotion_weights[emo] = emotion_weights.get(emo, 0) + conf
        
        return max(emotion_weights.items(), key=lambda x: x[1])[0]

    def preprocess_face(self, face_roi):
        """Optimized face preprocessing pipeline"""
        # Resize with anti-aliasing
        roi = cv.resize(face_roi, (48, 48), interpolation=cv.INTER_AREA)
        
        # Adaptive histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        roi = clahe.apply(roi)
        
        # Normalization and reshaping
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1)
        return np.expand_dims(roi, axis=0)

    def detect_faces(self, frame):
        """Optimized face detection with adaptive parameters"""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Dynamically adjust detection parameters based on frame size
        height, width = gray.shape
        min_size = max(100, int(min(height, width) * 0.2))
        
        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(min_size, min_size),
            flags=cv.CASCADE_SCALE_IMAGE
        )

    def process_frame(self, frame):
        """Main processing pipeline for a single frame"""
        processing_start = time.time()
        faces = self.detect_faces(frame)
        largest_face = max(faces, key=lambda r: r[2]*r[3]) if len(faces) > 0 else None
        
        if largest_face is not None:
            x, y, w, h = largest_face
            
            try:
                # Extract and preprocess face ROI
                face_roi = cv.cvtColor(frame[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
                processed_face = self.preprocess_face(face_roi)
                
                # Predict emotion
                predictions = self.model.predict(processed_face, verbose=0)[0]
                confidence = np.max(predictions)
                label = self.emotions[np.argmax(predictions)]
                
                # Apply confidence threshold and smoothing
                if confidence > 0.65:  # Higher threshold for more reliability
                    self.last_prediction = self.get_smoothed_prediction(label, confidence)
                
                # Visualization
                self.draw_detection(frame, x, y, w, h, label, confidence)
                
            except Exception as e:
                print(f"Face processing error: {e}")

        # Update performance metrics
        processing_time = time.time() - processing_start
        self.avg_processing_time = (
            (self.avg_processing_time * self.frames_processed + processing_time) / 
            (self.frames_processed + 1)
        )
        self.frames_processed += 1
        
        return frame

    def draw_detection(self, frame, x, y, w, h, label, confidence):
        """Optimized visualization drawing"""
        # Main face rectangle
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Emotion label with confidence
        cv.putText(frame, f"{label} ({confidence:.2f})", 
                  (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, 
                  (255, 255, 255), 2, cv.LINE_AA)
        
        # Performance stats
        fps = self.frames_processed / (time.time() - self.start_time)
        cv.putText(frame, f"FPS: {fps:.1f} | Avg: {self.avg_processing_time*1000:.1f}ms", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 
                  (255, 255, 0), 1, cv.LINE_AA)

    def run(self):
        """Main detection loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv.imshow('Optimized Emotion Detection', processed_frame)
                
                # Exit on 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                
                self.frame_count += 1
                
        finally:
            self.cap.release()
            cv.destroyAllWindows()
            print(f"Final stats - FPS: {self.frames_processed/(time.time()-self.start_time):.1f}")
            print(f"Average processing time: {self.avg_processing_time*1000:.1f}ms")

# Run the detector
if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()
