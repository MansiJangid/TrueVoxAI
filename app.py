from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
import joblib
import speech_recognition as sr
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load("mlp_emotion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    """Extract audio features for emotion recognition"""
    try:
        y_data, sr = librosa.load(file_path, sr=None, mono=True)
        
        if len(y_data) == 0:
            return None
            
        features = np.hstack([
            np.mean(librosa.feature.zero_crossing_rate(y_data).T, axis=0),
            np.mean(librosa.feature.chroma_stft(y=y_data, sr=sr).T, axis=0),
            np.mean(librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=40).T, axis=0),
            np.mean(librosa.feature.melspectrogram(y=y_data, sr=sr).T, axis=0)
        ])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def recognize_emotion(file_path):
    """Recognize emotion in audio file"""
    features = extract_features(file_path)
    if features is None:
        return {"error": "Failed to extract audio features"}

    scaled_features = scaler.transform(features.reshape(1, -1))

    prediction = model.predict(scaled_features)
    emotion = label_encoder.inverse_transform(prediction)[0]

    probs = model.predict_proba(scaled_features)[0]
    emotion_probs = {emotion: float(prob) for emotion, prob in zip(label_encoder.classes_, probs)}

    emotion_map = {
        'ANG': 'Angry',
        'DIS': 'Disgusted',
        'FEA': 'Fearful',
        'HAP': 'Happy',
        'NEU': 'Neutral',
        'SAD': 'Sad'
    }
    
    full_emotion = emotion_map.get(emotion, emotion)
    
    return {
        "emotion": emotion,
        "emotion_name": full_emotion,
        "confidence": float(max(probs)),
        "probabilities": emotion_probs
    }

def transcribe_audio(file_path):
    """Transcribe speech to text from audio file"""
    recognizer = sr.Recognizer()
    
    try:

        with sr.AudioFile(file_path) as source:

            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return {"text": text}
    except sr.UnknownValueError:
        return {"text": "Speech unclear", "error": "Could not understand audio"}
    except sr.RequestError as e:
        return {"text": "", "error": f"Speech service error: {e}"}
    except Exception as e:
        return {"text": "", "error": f"Error transcribing audio: {e}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        emotion_result = recognize_emotion(file_path)
        transcription_result = transcribe_audio(file_path)

        result = {
            "filename": filename,
            "emotion": emotion_result,
            "transcription": transcription_result
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
    finally:
        pass

if __name__ == '__main__':
    app.run(debug=True)
