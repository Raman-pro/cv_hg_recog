import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load Data
print("Loading dataset...")
try:
    data = pd.read_csv('drone_dataset.csv')
except FileNotFoundError:
    print("Error: drone_dataset.csv not found. Run collect_data.py first.")
    exit()

# 2. Prepare Data
X = data.drop('label', axis=1) # Features (Coordinates)
y = data['label']              # Target (Labels)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save Model
with open('drone_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'drone_rf_model.pkl'")