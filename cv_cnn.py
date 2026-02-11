import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data Setup
BATCH_SIZE = 32
IMG_SIZE = (64, 64) # Resize images to 64x64 pixels
DATA_DIR = 'dataset'

# Automatically load and augment data (flip, zoom slightly to make model robust)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. Build the CNN Model
model = Sequential([
    # Layer 1: Convolution (Extract Features)
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2), # Reduce size
    
    # Layer 2: Convolution
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Layer 3: Flatten & Dense (Classification)
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    Dense(4, activation='softmax') # Output: 4 neurons for 4 gestures
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train
print("Starting Training...")
model.fit(train_generator, epochs=10, validation_data=val_generator)

# 4. Save the Model
model.save("drone_gesture_model.h5")
print("Model Saved!")