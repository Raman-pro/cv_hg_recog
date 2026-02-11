import cv2
import os
import time

# CONFIG
DATA_DIR = "dataset"
LABELS = ["takeoff", "land", "none"] # Change these to match your needs
IMAGES_PER_LABEL = 80

# Create directories
for label in LABELS:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

cap = cv2.VideoCapture(0)

print("Controls:")
print("Press 't' to collect TAKEOFF images")
print("Press 'l' to collect LAND images")
print("Press 'n' to collect NONE images")
print("Press 'q' to quit")

def collect(label):
    count = 0
    print(f"Collecting {label}...")
    while count < IMAGES_PER_LABEL:
        ret, frame = cap.read()
        if not ret: break
        
        # Flip to mirror (intuitive for user)
        frame = cv2.flip(frame, 1)
        
        # Save image
        timestamp = int(time.time() * 1000)
        # Unique filename
        filename = os.path.join(DATA_DIR, label, f"{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        
        # Visual Feedback
        cv2.putText(frame, f"Collecting {label}: {count}/{IMAGES_PER_LABEL}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Data Collector", frame)
        
        count += 1
        cv2.waitKey(50) # 50ms delay between shots

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    cv2.putText(frame, "Press t, l, n to record", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Data Collector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('t'): collect("takeoff")
    if key == ord('l'): collect("land")
    if key == ord('n'): collect("none")

cap.release()
cv2.destroyAllWindows()