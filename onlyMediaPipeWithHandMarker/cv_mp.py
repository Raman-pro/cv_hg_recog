import mediapipe as mp
import cv2
import time
import numpy as np
from Models.Drone import Drone
drone=Drone()
def get_rotated_triangle(center, size, yaw_deg):
    """Calculates points for an isoceles triangle rotated by yaw."""
    pts = np.array([
        [0, -size],
        [-size//1.5, size],
        [size//1.5, size]
    ])

    angle = np.radians(yaw_deg)
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated_pts = pts @ rot_matrix.T
    final_pts = (rotated_pts + center).astype(np.int32)
    return final_pts

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # Draw Control Zones
        left_center = (w // 4, h // 2)
        right_center = (w * 3 // 4, h // 2)
        cv2.circle(frame, left_center, 290, (100, 100, 100), 2)
        cv2.circle(frame, left_center, 40, (100, 100, 100), 2)
        cv2.circle(frame, right_center, 290, (100, 100, 100), 2)
        cv2.circle(frame, right_center, 40, (100, 100, 100), 2)
        
                # --- GRAPHICAL VISUALIZATION ---
        
        # 1. Create Display Windows
        top_view = np.zeros((800, 800, 3), dtype=np.uint8)
        side_view = np.zeros((400, 200, 3), dtype=np.uint8)

        # 2. Draw Top View (X, Y, Yaw)
        # Using placeholder values if drone class isn't updating yet
        # Replace 0, 0, 0 with drone.x, drone.y, drone.yaw
        d_center = (400, 400)
        # We scale values to fit the 400x400 window
        tri_pts = get_rotated_triangle((d_center[0]+drone.x,d_center[1]+drone.y), 20, drone.yaw) # Use drone.yaw here
        cv2.drawContours(top_view, [tri_pts], 0, (0, 255, 255), -1)
        cv2.putText(top_view, f"Top View: X={int(drone.x)}, Y={int(drone.y)}, Yaw={int(drone.yaw)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # 3. Draw Side View (Z)
        # Replace 0 with drone.z
        z_pos = 200 - int(drone.z * 0.5) 
        cv2.fillPoly(side_view, [np.array([[100, z_pos-10], [70, z_pos+10], [130, z_pos+10]])], (255, 0, 255))
        cv2.putText(side_view, f"Side View: Z={int(drone.z)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Show everything

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        
        if latest_result and latest_result.hand_landmarks:
            left=0
            right=0
            for idx, landmarks in enumerate(latest_result.hand_landmarks):
                trigger_cord = [int(landmarks[8].x * w), int(landmarks[8].y * h)]
                
                # Logic for Left Hand (idx=1) vs Right Hand (idx=0)
                # Note: MediaPipe idx can swap; usually better to check handedness label
                x,y,z,yaw = 0,0,0,0 # Placeholder values for drone state updates
                if (idx == 1):
                    left=trigger_cord
                    dist_centre = ((trigger_cord[0]-left_center[0])**2 + (trigger_cord[1]-left_center[1])**2)**0.5
                    if dist_centre > 290:
                        trigger_cord[0] = int(left_center[0] + (trigger_cord[0]-left_center[0]) * 290 / dist_centre)
                        trigger_cord[1] = int(left_center[1] + (trigger_cord[1]-left_center[1]) * 290 / dist_centre)
                    if not dist_centre<40:
                        
                        z=-(trigger_cord[1] - left_center[1])
                        yaw=(trigger_cord[0] - left_center[0])
                    # Update Drone Z and Yaw
                    # drone.z = -(trigger_cord[1] - left_center[1]) 
                    # drone.yaw = (trigger_cord[0] - left_center[0])
                else:
                    dist_centre = ((trigger_cord[0]-right_center[0])**2 + (trigger_cord[1]-right_center[1])**2)**0.5
                    if dist_centre > 290:
                        trigger_cord[0] = int(right_center[0] + (trigger_cord[0]-right_center[0]) * 290 / dist_centre)
                        trigger_cord[1] = int(right_center[1] + (trigger_cord[1]-right_center[1]) * 290 / dist_centre)
                    if not dist_centre<40:
                        x = (trigger_cord[0] - right_center[0])
                        y=(trigger_cord[1] - right_center[1])
                drone.update_position(x*0.01, y*0.01, z*0.01, yaw*0.01) # Update drone state with new values    
                    # Update Drone X and Y
                    # drone.x = (trigger_cord[0] - right_center[0])
                    # drone.y = -(trigger_cord[1] - right_center[1])

                cv2.circle(frame, (trigger_cord[0], trigger_cord[1]), 10, (0, 255, 0), -1)
        cv2.imshow('Drone Controller', frame)
        cv2.imshow('Top View', top_view)
        cv2.imshow('Side View', side_view)


cap.release()
cv2.destroyAllWindows()