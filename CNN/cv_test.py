import cv2
import numpy as np
import tensorflow as tf
# from djitellopy import Tello

# 1. Load Model
model = tf.keras.models.load_model("drone_gesture_model.h5")
CLASSES = [ 'left', 'right', 'stop', 'up'] # Alphabetical order from flow_from_directory

# # 2. Setup Drone
# drone = Tello()
# drone.connect()
# drone.streamon()
# drone.takeoff() # Uncomment when ready to fly!

cap = cv2.VideoCapture(0) # Use webcam for control

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Get the ROI (Same as training)
    cv2.rectangle(frame, (200,200), (600, 600), (0, 255, 0), 2)
    roi = frame[200:600, 200:600] 

    
    # Preprocess ROI for Model
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0) # Add batch dimension
    
    # Predict
    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    command = CLASSES[index]
    
    # Display Command
    text = f"{command} ({int(confidence*100)}%)"
    cv2.putText(frame, text, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Send to Drone (Only if confident)
    if confidence > 0.8:
        if command == 'up': print('up')
        elif command == 'left': print('left')
        elif command == 'right': print('right')
        elif command == 'stop': print('stop')
    else:
        print('hover')

    cv2.imshow("Drone Commander", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()