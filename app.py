import os
import cv2
import uuid
import json
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import mediapipe as mp
from flask_session import Session

# Configuring logger
logging.basicConfig(level=logging.DEBUG)

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Mediapipe Hand detection model initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route: Home page
@app.route('/')
def index():
    if not session.get('session_id'):
        session['session_id'] = str(uuid.uuid4())  # Create a session identifier
    logging.info(f"Session ID: {session['session_id']}")
    return render_template('index.html')

# Route: Upload and encode sign language from video file
@app.route('/encode', methods=['POST'])
def encode():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'No video selected for uploading'}), 400

    if not allowed_file(video_file.filename):
        return jsonify({'error': 'Unsupported file type. Allowed types are: mp4, avi, mov, webm'}), 400

    # Secure the filename and save the uploaded file
    filename = secure_filename(video_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(file_path)

    logging.info(f"Video saved to {file_path}")

    # Initialize video capture
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing {frame_count} frames")

    # Process each frame and extract hand landmarks
    frame_list = []
    frame_index = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"End of video or read error at frame {frame_index}")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    frame_list.append({'frame_index': frame_index, 'landmarks': landmarks})

            frame_index += 1

        cap.release()

        # Save the result to a session-based JSON file for later retrieval
        session_id = session.get('session_id')
        output_file = f"static/results/{session_id}_landmarks.json"
        with open(output_file, 'w') as outfile:
            json.dump(frame_list, outfile)

        logging.info(f"Landmarks saved to {output_file}")

        return jsonify({'message': 'Hand landmarks processed', 'landmarks_url': output_file})

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the video'}), 500

# Route: Real-time webcam processing (Complex Endpoint)
@app.route('/real_time_encode', methods=['POST'])
def real_time_encode():
    try:
        cap = cv2.VideoCapture(0)
        logging.info("Webcam stream started")

        real_time_landmarks = []
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Frame capture failed at index {frame_index}")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    real_time_landmarks.append({'frame_index': frame_index, 'landmarks': landmarks})

            frame_index += 1
            if frame_index > 100:  # Limiting to 100 frames for simplicity
                break

        cap.release()

        # Save to session-specific file
        session_id = session.get('session_id')
        real_time_file = f"static/results/{session_id}_real_time_landmarks.json"
        with open(real_time_file, 'w') as outfile:
            json.dump(real_time_landmarks, outfile)

        logging.info(f"Real-time landmarks saved to {real_time_file}")

        return jsonify({'message': 'Real-time hand landmarks processed', 'landmarks_url': real_time_file})

    except Exception as e:
        logging.error(f"Error in real-time processing: {str(e)}")
        return jsonify({'error': 'An error occurred during real-time processing'}), 500

# Route: Get encoded landmarks
@app.route('/results/<session_id>')
def get_results(session_id):
    try:
        result_file = f"static/results/{session_id}_landmarks.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as infile:
                data = json.load(infile)
            return jsonify({'landmarks': data})
        else:
            return jsonify({'error': 'No results found for the given session ID'}), 404
    except Exception as e:
        logging.error(f"Error retrieving results: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving results'}), 500

if __name__ == '__main__':
    app.run(debug=True)
