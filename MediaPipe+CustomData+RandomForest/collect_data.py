import cv2
import mediapipe as mp
import csv
import time
import os

# --- CONFIGURATION ---
CSV_FILE = 'drone_dataset.csv'

# YOUR CUSTOM LABELS MAPPING
# Note: I matched this exactly to what you provided in your snippet
LABELS = {
    ord('t'): 'takeoff', 
    ord('l'): 'land', 
    ord('a'): 'anticlockwise', 
    ord('c'): 'clockwise', 
    ord('n'): 'hover' # You mapped 'n' to hover/none
}

# --- SETUP MEDIAPIPE ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'coord_{i}' for i in range(42)]
        writer.writerow(header)

# --- INITIALIZE LANDMARKER ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, 
    num_hands=1
)

print("Attempting to open camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")
print("=== CONTROLS ===")
print("  't' -> Set label to TAKEOFF")
print("  'l' -> Set label to LAND")
print("  'a' -> Set label to ANTICLOCKWISE")
print("  'c' -> Set label to CLOCKWISE")
print("  'n' -> Set label to HOVER")
print("  's' -> SAVE current frame")
print("  'q' -> QUIT")
print("================")

current_label = "hover" # Default start label

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        
        # Variable to hold landmarks for this specific frame
        current_landmarks_data = None 

        try:
            # Run detection
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # If hand detected, draw it and store data
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # Draw points
                    h, w, _ = frame.shape
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    
                    # Store data for saving later
                    # We only take the first hand detected
                    current_landmarks_data = hand_landmarks

        except Exception as e:
            print(f"MediaPipe Error: {e}")

        # --- UI OVERLAY ---
        # Draw black box for text
        cv2.rectangle(frame, (0,0), (350, 80), (0,0,0), -1)
        
        # Show Current Label
        # Highlight green if 'takeoff/land', yellow for others
        color = (0, 255, 0) if current_label in ['takeoff', 'land'] else (0, 255, 255)
        cv2.putText(frame, f"Label: {current_label.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Press keys to switch or 's' to save", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Data Collector', frame)
        
        # --- KEYBOARD LOGIC (THE FIX) ---
        # This is now OUTSIDE the detection block, running every frame
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        # Check if key is in your LABELS dictionary
        elif key in LABELS:
            current_label = LABELS[key]
            print(f"-> Switched label to: {current_label}")
        
        # Save only if 's' is pressed AND we have data
        elif key == ord('s'):
            if current_landmarks_data is not None:
                # Flatten data
                row = []
                for lm in current_landmarks_data:
                    row.extend([lm.x, lm.y])
                
                # Write to CSV
                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_label] + row)
                
                print(f"Saved 1 sample for {current_label}")
                
                # Visual Flash Effect
                cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 20)
                cv2.imshow('Data Collector', frame)
                cv2.waitKey(50) # Small pause to show the flash
            else:
                print("Cannot save: No hand detected!")

cap.release()
cv2.destroyAllWindows()