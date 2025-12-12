import kagglehub
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ CONFIGURATION PARAMETERS
# ==========================================
MODEL_NAME = 'letter_recognition_cnn.h5'
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
SEED = 42
# ==========================================

def get_dataset_path():
    """Downloads dataset via kagglehub and finds the csv file path."""
    print("⬇️ Downloading dataset via KaggleHub...")
    path = kagglehub.dataset_download("sachinpatel21/az-handwritten-alphabets-in-csv-format")
    print(f"✅ Dataset downloaded to: {path}")
    
    # Search for the .csv file in the downloaded folder
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("Could not find a CSV file in the downloaded dataset folder.")
    
    return csv_files[0]

def check_gpu():
    """Configures GPU memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Detected: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU detected. Training will run on CPU.")

def load_and_process_data(csv_path):
    print(f"📖 Reading data from {csv_path}...")
    # Read CSV
    data = pd.read_csv(csv_path).astype('float32')
    
    # Split features and labels (Col 0 is label, rest are pixels)
    X = data.drop('0', axis=1)
    y = data['0']
    
    # Clean memory
    del data

    # Reshape from rows to 28x28 images
    X = np.array(X)
    X = X.reshape(X.shape[0], 28, 28, 1)
    
    # Resize images to 32x32 using OpenCV
    print("🔄 Resizing images to 32x32 for architecture compatibility...")
    X_resized = np.zeros((X.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    
    for i in range(X.shape[0]):
        # Convert to uint8 for opencv, resize, then back to float
        img = X[i].astype('uint8')
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_resized[i] = img.reshape(IMG_SIZE, IMG_SIZE, 1)
        
    # Normalize pixel values (0 to 1)
    X_resized = X_resized / 255.0
    
    # One-hot encode labels (26 classes)
    y = to_categorical(y, num_classes=26)
    
    # Split data
    return train_test_split(X_resized, y, test_size=TEST_SPLIT, random_state=SEED)

def build_cnn_model():
    """Builds the 4-block CNN architecture."""
    model = Sequential()
    
    # Define Input explicitly
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

    # --- Block 1 ---
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output: 16x16

    # --- Block 2 ---
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output: 8x8

    # --- Block 3 ---
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output: 4x4

    # --- Block 4 ---
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output: 2x2

    # --- Classifier ---
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax')) # Output layer (A-Z)

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    check_gpu()
    
    # 1. Get Data
    try:
        csv_file_path = get_dataset_path()
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

    # 2. Process Data
    X_train, X_test, y_train, y_test = load_and_process_data(csv_file_path)
    print(f"📊 Training Data Shape: {X_train.shape}")
    
    # 3. Build Model
    model = build_cnn_model()
    model.summary()

    # 4. Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1
    )
    datagen.fit(X_train)

    # 5. Train
    print("🚀 Starting Training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 6. Save
    model.save(MODEL_NAME)
    print(f"💾 Model saved successfully as: {MODEL_NAME}")