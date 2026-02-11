import pickle
import numpy as np

class GestureModel:
    def __init__(self, model_path):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}.")
            self.model = None

    def predict(self, landmarks):
        """
        Predicts gesture from landmarks.
        landmarks: list of (x, y) coordinates or flattened list
        """
        if self.model is None:
            return None, 0.0

        # Flatten if necessary (assuming input is list of objects with x, y)
        if hasattr(landmarks[0], 'x'):
             row = []
             for lm in landmarks:
                 row.extend([lm.x, lm.y])
        else:
            row = landmarks

        prediction = self.model.predict([row])[0]
        probability = self.model.predict_proba([row]).max()
        return prediction, probability
