import mediapipe as mp
import time
import cv2

class HandTracker:
    def __init__(self, model_path, num_hands=2, min_detection_confidence=0.5, min_presence_confidence=0.5, min_tracking_confidence=0.5):
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_result = None

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self.print_result
        )
        
        self.landmarker = self.HandLandmarker.create_from_options(options)

    def print_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def detect_async(self, frame):
        # MediaPipe expects RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, timestamp)

    def get_latest_result(self):
        return self.latest_result

    def close(self):
        self.landmarker.close()
