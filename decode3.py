import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import argparse

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Argument parsing for command line execution
parser = argparse.ArgumentParser(description="Decode sign language gesture from video.")
parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
parser.add_argument('--model_type', type=str, choices=['tf_saved_model', 'tflite'], default='tf_saved_model',
                    help='Type of model to use: "tf_saved_model" or "tflite".')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
args = parser.parse_args()

# Helper function to extract hand landmarks from a video
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
            landmarks_sequence.append([[0, 0, 0]] * 21)  # If no hand detected, zero out the frame

    cap.release()

    return np.array(landmarks_sequence)

# TensorFlow SavedModel-based prediction
def predict_with_tf_saved_model(model_path, landmarks):
    # Load the TensorFlow model
    model = tf.keras.models.load_model(model_path)
    
    # Expand dimensions to match the model input shape
    landmarks = np.expand_dims(landmarks, axis=0)
    
    # Make prediction
    prediction = model.predict(landmarks)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return predicted_class

# TensorFlow Lite-based prediction
def predict_with_tflite_model(model_path, landmarks):
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare the input data (expand dimensions to match the model input shape)
    input_data = np.expand_dims(landmarks, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the predicted output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)[0]
    
    return predicted_class

# Main decoding function
def decode_sign_language(video_path, model_type, model_path):
    # Extract landmarks from the video
    landmarks = extract_landmarks_from_video(video_path)
    print(f"Extracted landmarks shape: {landmarks.shape}")

    # Predict using the chosen model type
    if model_type == 'tf_saved_model':
        predicted_class = predict_with_tf_saved_model(model_path, landmarks)
    elif model_type == 'tflite':
        predicted_class = predict_with_tflite_model(model_path, landmarks)
    
    print(f"Predicted class: {predicted_class}")
    return predicted_class

# Run the decoding
if __name__ == '__main__':
    decode_sign_language(args.video, args.model_type, args.model_path)
