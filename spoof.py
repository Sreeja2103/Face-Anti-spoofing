import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set up paths and parameters
# ... (previous imports)
import cv2  # Add this

def extract_video_frames(video_path, output_dir, frames_to_save=10):
    """Extract frames from video files"""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    
    while success and count < frames_to_save:
        frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1

# Process videos first
dataset_path = r"C:\Users\21311\Major Project1\dataset"

print("Processing video files...")
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith('.mp4'):
            video_path = os.path.join(root, file)
            print(f"Extracting frames from: {video_path}")
            extract_video_frames(video_path, root)

# Then create dataframe
data = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_lower = file.lower()
        if file_lower.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file)
            
            # Flexible label detection
            if 'live_selfie' in file_lower:
                label = 'real'
            elif any(x in file_lower for x in ['live_video', 'frame_']):  # Frames from videos
                label = 'spoof'
            else:
                print(f"Skipping unclassified file: {file}")
                continue
                
            data.append((path, label))

# ... (rest of your original code)
df = pd.DataFrame(data, columns=['filename', 'class'])

# Check class distribution
print("Class distribution:")
print(df['class'].value_counts())

# Verify we have 2 classes
if len(df['class'].unique()) != 2:
    raise ValueError("Dataset must contain both 'real' and 'spoof' classes")

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['class'], random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)
image_size = (224, 224)
batch_size = 32
epochs = 20
learning_rate = 0.0001
# Create data generators
def create_generator(datagen, dataframe):
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='filename',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['real', 'spoof']
    )

train_generator = create_generator(train_datagen, train_df)
val_generator = create_generator(val_test_datagen, val_df)
test_generator = create_generator(val_test_datagen, test_df)

# Rest of the model code remains the same as previous version
# ... [Keep the model building, training, and evaluation code unchanged]
# Calculate class weights
class_counts = train_df['class'].value_counts()
class_weights = {0: class_counts.sum()/(2*class_counts[0]), 
                 1: class_counts.sum()/(2*class_counts[1])}

# Build model
base_model = MobileNetV2(weights='imagenet', 
                        include_top=False, 
                        input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5',
                            monitor='val_accuracy',
                            save_best_only=True,
                            mode='max')

early_stop = EarlyStopping(monitor='val_loss',
                          patience=5,
                          restore_best_weights=True)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# Evaluate model
model.load_weights('best_model.h5')  # Load best weights

test_results = model.evaluate(test_generator)
print(f'Test Accuracy: {test_results[1]*100:.2f}%')

# Generate predictions
y_pred = model.predict(test_generator)
y_pred = (y_pred > 0.5).astype(int)
y_true = test_generator.classes

# Classification report
print(classification_report(y_true, y_pred, target_names=['spoof', 'real']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save final model
model.save('face_anti_spoofing_model.h5')