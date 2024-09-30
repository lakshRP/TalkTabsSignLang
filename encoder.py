import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten, TimeDistributed, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                landmarks_sequence.append(landmarks)
        else:
            # If no hand is detected, append a zeroed-out frame
            landmarks_sequence.append([[0, 0, 0]] * 21)

    cap.release()
    
    return np.array(landmarks_sequence)

# Example: Extract landmarks from a video
video_path = 'path_to_video_file.mp4'
landmarks = extract_landmarks_from_video(video_path)
print(f"Extracted landmarks shape: {landmarks.shape}")

# Path to dataset of sign language videos
DATASET_PATH = 'path_to_sign_language_dataset'

# Dictionary to store the gesture names and their corresponding landmark sequences
data = []
labels = []

# Loop through all the folders corresponding to different gestures
for gesture_label in os.listdir(DATASET_PATH):
    gesture_folder = os.path.join(DATASET_PATH, gesture_label)
    if os.path.isdir(gesture_folder):
        for video_file in os.listdir(gesture_folder):
            video_path = os.path.join(gesture_folder, video_file)
            landmarks = extract_landmarks_from_video(video_path)
            data.append(landmarks)
            labels.append(gesture_label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

# One-hot encode the labels
lb = LabelBinarizer()
labels_encoded = lb.fit_transform(labels)
print(f"Encoded labels shape: {labels_encoded.shape}")

def create_sign_language_model(input_shape, num_classes):
    model = Sequential()

    # TimeDistributed CNN layers
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Flatten()))

    # LSTM layers to capture temporal dependencies
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))

    # Fully connected layer and output layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Example: Create the model
input_shape = (None, 21, 3)  # Example input shape: Sequence


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_sign_language_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_accuracy', mode='max', restore_best_weights=True)
    ]
)

# Save the trained model
model.save('final_sign_language_model.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")
def predict_sign_language_gesture(model, video_path):
    # Extract landmarks from the video
    landmarks = extract_landmarks_from_video(video_path)

    # Reshape the input to fit the model's expected input shape
    landmarks = np.expand_dims(landmarks, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(landmarks)
    predicted_class = np.argmax(prediction, axis=1)

    # Return the predicted gesture
    return lb.classes_[predicted_class[0]]

# Example: Predict gesture for a new video
new_video_path = 'path_to_new_video.mp4'
predicted_gesture = predict_sign_language_gesture(model, new_video_path)
print(f"Predicted sign language gesture: {predicted_gesture}")
import random

def augment_landmarks(landmarks, augmentations=3):
    augmented_data = []

    for _ in range(augmentations):
        # Random horizontal flip (flipping landmarks along x-axis)
        if random.choice([True, False]):
            flipped_landmarks = np.copy(landmarks)
            for frame in flipped_landmarks:
                frame[:, 0] = 1 - frame[:, 0]  # Flip x-coordinate
            augmented_data.append(flipped_landmarks)

        # Time warping (shifting frames forward/backward)
        if random.choice([True, False]):
            shift = random.randint(-5, 5)
            shifted_landmarks = np.roll(landmarks, shift, axis=0)
            augmented_data.append(shifted_landmarks)

        # Add Gaussian noise to landmarks
        if random.choice([True, False]):
            noisy_landmarks = landmarks + np.random.normal(0, 0.01, landmarks.shape)
            augmented_data.append(noisy_landmarks)

        # Random frame dropout
        if random.choice([True, False]):
            dropout_landmarks = np.copy(landmarks)
            for i in range(len(dropout_landmarks)):
                if random.random() < 0.1:
                    dropout_landmarks[i, :, :] = 0  # Drop out a frame by setting all coordinates to 0
            augmented_data.append(dropout_landmarks)

    return augmented_data

# Augment the dataset with the new function
augmented_data = []
augmented_labels = []
for i in range(len(data)):
    augmented_data.append(data[i])
    augmented_labels.append(labels[i])
    
    augmented_samples = augment_landmarks(data[i])
    for sample in augmented_samples:
        augmented_data.append(sample)
        augmented_labels.append(labels[i])

# Convert augmented data to numpy arrays
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)
print(f"Augmented data shape: {augmented_data.shape}, Augmented labels shape: {augmented_labels.shape}")

class PerformanceLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_interval=5):
        super().__init__()
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            logging.info(f"Epoch {epoch + 1}: Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Val Loss: {logs['val_loss']}, Val Accuracy: {logs['val_accuracy']}")

class PredictionVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, sample_video, log_interval=10):
        super().__init__()
        self.sample_video = sample_video
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            prediction = self.model.predict(np.expand_dims(self.sample_video, axis=0))
            predicted_class = np.argmax(prediction, axis=1)
            logging.info(f"Epoch {epoch + 1}: Predicted gesture: {lb.classes_[predicted_class[0]]}")

# Instantiate the custom callbacks
sample_video_path = 'path_to_sample_video.mp4'
sample_landmarks = extract_landmarks_from_video(sample_video_path)

performance_logger = PerformanceLogger(log_interval=5)
prediction_visualizer = PredictionVisualizer(sample_landmarks)
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform

# Define a search space for hyperparameters
param_grid = {
    'learning_rate': uniform(0.0001, 0.01),
    'batch_size': [16, 32, 64],
    'dropout_rate': [0.3, 0.5, 0.7],
    'lstm_units': [128, 256, 512],
    'cnn_filters': [32, 64, 128],
}

param_list = list(ParameterSampler(param_grid, n_iter=10))

best_model = None
best_val_acc = 0

# Random Search: Train models with randomly sampled hyperparameters
for params in param_list:
    logging.info(f"Training with params: {params}")
    
    # Create a new model with the sampled hyperparameters
    model = Sequential()
    model.add(TimeDistributed(Conv2D(params['cnn_filters'], (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(params['lstm_units'], return_sequences=False))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile and train the model
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=params['batch_size'],
        verbose=0,
        callbacks=[performance_logger]
    )

    # Evaluate model
    val_acc = max(history.history['val_accuracy'])
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model

logging.info(f"Best validation accuracy: {best_val_acc}")
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the MobileNetV2 model with pre-trained weights, excluding the top layers
mobilenet_base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the MobileNetV2 base layers
for layer in mobilenet_base.layers:
    layer.trainable = False

def create_transfer_learning_model(input_shape, num_classes):
    model = Sequential()

    # Add TimeDistributed wrapper for MobileNetV2
    model.add(TimeDistributed(mobilenet_base, input_shape=input_shape))
    model.add(TimeDistributed(Flatten()))

    # LSTM for temporal dependency
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))

    # Fully connected layer and output
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Create and compile the model
transfer_learning_model = create_transfer_learning_model(input_shape=(None, 224, 224, 3), num_classes=num_classes)
transfer_learning_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with MobileNetV2 base
history = transfer_learning_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[performance_logger, prediction_visualizer]
)
# Save the model for TensorFlow Serving deployment
transfer_learning_model.save('sign_language_saved_model')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('sign_language_saved_model')
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)
