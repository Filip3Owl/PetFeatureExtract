import os
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Configuration Constants ---
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
EPOCHS = 20
SEED = 123
# Adjust this path to your local directory structure
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'PetImages')

def clean_dataset(directory):
    """
    Scans the dataset directory and removes corrupted or non-JPEG images 
    to prevent training crashes.
    """
    print("Starting dataset cleaning...")
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(directory, folder_name)
        if not os.path.exists(folder_path):
            continue
            
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "rb") as fobj:
                    # Check for JFIF identifier in the header
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                if not is_jfif:
                    num_skipped += 1
                    os.remove(fpath)
            except (IOError, OSError):
                num_skipped += 1
                if os.path.exists(fpath):
                    os.remove(fpath)
                    
    print(f"Cleaning complete. Removed {num_skipped} invalid files.")

def build_lenet5_adapted(input_shape=(64, 64, 3), num_classes=1):
    """
    Builds an adapted LeNet-5 architecture for RGB image classification.
    """
    model = models.Sequential([
        # C1: Convolutional Layer (6 filters, 5x5 kernel)
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        
        # S2: Max Pooling Layer (Downsampling)
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # C3: Convolutional Layer (16 filters, 5x5 kernel)
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        
        # S4: Max Pooling Layer
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flattening 2D maps into 1D vectors for Dense layers
        layers.Flatten(),
        
        # F5: Fully Connected Layer (120 neurons)
        layers.Dense(120, activation='relu'),
        
        # F6: Fully Connected Layer (84 neurons)
        layers.Dense(84, activation='relu'),
        
        # Output Layer: Sigmoid for Binary Classification (Dog vs Cat)
        layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

def main():
    # 1. Prepare and Clean Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' not found.")
        return

    clean_dataset(DATASET_PATH)

    # 2. Load Datasets
    print("\nLoading training and validation sets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 3. Data Pre-processing & Performance Optimization
    # Rescaling pixels from [0, 255] to [0, 1]
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Optimization using Prefetching
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    # 4. Initialize and Compile Model
    model = build_lenet5_adapted(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 5. Training
    print("\nStarting training process...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print("\nTraining finished successfully.")

if __name__ == "__main__":
    main()