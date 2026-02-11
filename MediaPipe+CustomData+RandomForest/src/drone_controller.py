import cv2
import time
from .utils.hand_tracker import HandTracker
from .utils.visualization import Visualization
from .models.drone import Drone
from .models.gesture_model import GestureModel
from . import config

class DroneController:
    def __init__(self):
        self.drone = Drone()
        self.hand_tracker = HandTracker(
            model_path=config.MP_MODEL_PATH,
            num_hands=config.NUM_HANDS,
            min_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.gesture_model = GestureModel(config.RF_MODEL_PATH)
        self.viz = Visualization(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        self.running = True

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            # Process Frame
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Detect Hands
            self.hand_tracker.detect_async(frame)
            
            # Draw Control Zones
            right_center = (w * 3 // 4, h // 2)
            self.viz.draw_control_zones(frame, w, h, right_center, 
                                      config.CONTROL_ZONE_RADIUS_LARGE, 
                                      config.CONTROL_ZONE_RADIUS_SMALL)

            # Process Results
            latest_result = self.hand_tracker.get_latest_result()
            if latest_result and latest_result.hand_landmarks:
                for idx, landmarks in enumerate(latest_result.hand_landmarks):
                     # Logic from cv_mp.py
                    trigger_cord = [int(landmarks[8].x * w), int(landmarks[8].y * h)]
                    x, y, z, yaw = 0, 0, 0, 0
                    
                    if idx == 1: # Left Hand (per original code assumption)
                        prediction, probability = self.gesture_model.predict(landmarks)
                        
                        if probability > config.PROBABILITY_THRESHOLD:
                            if prediction == 'takeoff':
                                z = 2
                            elif prediction == 'land':
                                z = -2
                            elif prediction == 'anticlockwise':
                                yaw = -2
                            elif prediction == 'clockwise':
                                yaw = 2
                    else: # Right Hand (per original code assumption)
                        # Detection of distance from center
                        dist_centre = ((trigger_cord[0]-right_center[0])**2 + (trigger_cord[1]-right_center[1])**2)**0.5
                        
                        if dist_centre > config.CONTROL_ZONE_RADIUS_LARGE:
                            ratio = config.CONTROL_ZONE_RADIUS_LARGE / dist_centre
                            trigger_cord[0] = int(right_center[0] + (trigger_cord[0]-right_center[0]) * ratio)
                            trigger_cord[1] = int(right_center[1] + (trigger_cord[1]-right_center[1]) * ratio)
                        
                        if not dist_centre < config.CONTROL_ZONE_RADIUS_SMALL:
                             x = (trigger_cord[0] - right_center[0])
                             y = (trigger_cord[1] - right_center[1])
                    
                    # Update Drone
                    self.drone.update_position(x * 0.01, y * 0.01, z, yaw)
                    
                    # Draw Finger Point
                    cv2.circle(frame, (trigger_cord[0], trigger_cord[1]), 10, (0, 255, 0), -1)

            # Draw Visualization Windows
            top_view = self.viz.draw_top_view(self.drone)
            side_view = self.viz.draw_side_view(self.drone)

            # Show Windows
            cv2.imshow('Drone Controller', frame)
            cv2.imshow('Top View', top_view)
            cv2.imshow('Side View', side_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.hand_tracker.close()
        cv2.destroyAllWindows()
