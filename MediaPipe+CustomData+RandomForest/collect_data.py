import cv2
import mediapipe as mp
import csv
import time
import os
import src.config as config

class DataCollector:
    def __init__(self):
        self.labels = {
            ord('t'): 'takeoff', 
            ord('l'): 'land', 
            ord('a'): 'anticlockwise', 
            ord('c'): 'clockwise', 
            ord('n'): 'hover'
        }
        self.current_label = "hover"
        
        self.init_csv()
        self.init_mediapipe()
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)

    def init_csv(self):
        if not os.path.exists(config.DATASET_PATH):
            with open(config.DATASET_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['label'] + [f'coord_{i}' for i in range(42)]
                writer.writerow(header)

    def init_mediapipe(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.MP_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO, 
            num_hands=1
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("=== CONTROLS ===")
        print("  't' -> Set label to TAKEOFF")
        print("  'l' -> Set label to LAND")
        print("  'a' -> Set label to ANTICLOCKWISE")
        print("  'c' -> Set label to CLOCKWISE")
        print("  'n' -> Set label to HOVER")
        print("  's' -> SAVE current frame")
        print("  'q' -> QUIT")
        print("================")

        with self.landmarker:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                
                current_landmarks_data = None 

                try:
                    result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                    if result.hand_landmarks:
                        for hand_landmarks in result.hand_landmarks:
                            self.draw_landmarks(frame, hand_landmarks)
                            current_landmarks_data = hand_landmarks
                except Exception as e:
                    print(f"MediaPipe Error: {e}")

                self.draw_ui(frame)
                cv2.imshow('Data Collector', frame)
                
                if not self.handle_input(frame, current_landmarks_data):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    def draw_ui(self, frame):
        cv2.rectangle(frame, (0,0), (350, 80), (0,0,0), -1)
        color = (0, 255, 0) if self.current_label in ['takeoff', 'land'] else (0, 255, 255)
        cv2.putText(frame, f"Label: {self.current_label.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Press keys to switch or 's' to save", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def handle_input(self, frame, landmarks_data):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key in self.labels:
            self.current_label = self.labels[key]
            print(f"-> Switched label to: {self.current_label}")
        elif key == ord('s'):
            self.save_data(frame, landmarks_data)
        return True

    def save_data(self, frame, landmarks_data):
        if landmarks_data is not None:
            row = []
            for lm in landmarks_data:
                row.extend([lm.x, lm.y])
            
            with open(config.DATASET_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.current_label] + row)
            
            print(f"Saved 1 sample for {self.current_label}")
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 20)
            cv2.imshow('Data Collector', frame)
            cv2.waitKey(50)
        else:
            print("Cannot save: No hand detected!")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()