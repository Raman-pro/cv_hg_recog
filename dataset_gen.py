import cv2
import os

# 1. Setup Folders
labels = ['up', 'down', 'left', 'right', 'stop']
BASE_DIR = 'dataset'

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    for label in labels:
        os.makedirs(os.path.join(BASE_DIR, label))

cap = cv2.VideoCapture(0)
print("Controls: 'u'=Up, 'd'=Down, 'l'=Left, 'r'=Right, 's'=Stop, 'q'=Quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame for easier interaction
    frame = cv2.flip(frame, 1)
    
    # Define a Region of Interest (ROI) - Box where you put your hand
    # We only save the box content to make training faster/easier
    cv2.rectangle(frame, (200,200), (600, 600), (0, 255, 0), 2)
    roi = frame[200:600, 200:600] 
    
    cv2.imshow("Collector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Save Logic
    save_path = None
    if key == ord('u'): save_path = os.path.join(BASE_DIR, 'up')
    elif key == ord('d'): save_path = os.path.join(BASE_DIR, 'down')
    elif key == ord('l'): save_path = os.path.join(BASE_DIR, 'left')
    elif key == ord('r'): save_path = os.path.join(BASE_DIR, 'right')
    elif key == ord('s'): save_path = os.path.join(BASE_DIR, 'stop')
    elif key == ord('q'): break
    
    if save_path:
        # Generate unique filename
        count = len(os.listdir(save_path))
        filename = f"{save_path}/img_{count}.jpg"
        cv2.imwrite(filename, roi)
        print(f"Saved to {filename}")

cap.release()
cv2.destroyAllWindows()