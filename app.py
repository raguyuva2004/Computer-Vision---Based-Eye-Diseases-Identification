from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import sqlite3
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

# Initialize Flask
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
def get_db():
    conn = sqlite3.connect('predictions.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    return conn

# Class labels (MUST match model training order)
CLASS_LABELS = ["DiabeticRetinopathy","Glaucoma","Healthy","Myopia","ODIR-5K","cataract"] 

# Load model with verification
try:
    model = tf.keras.models.load_model("eye_disease_model.h5")
    print("✅ Model loaded successfully. Input shape:", model.input_shape)
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path):
    """Preprocess and predict image"""
    try:
        # Load and preprocess image (must match training)
        img = image.load_img(img_path, target_size=(128, 128))  # Your training size
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        preds = model.predict(img_array, verbose=0)[0]
        pred_class = CLASS_LABELS[np.argmax(preds)]
        confidence = float(np.max(preds))
        
        # Debug output
        print("🔍 Raw predictions:", {CLASS_LABELS[i]: float(preds[i]) for i in range(6)})
        
        return pred_class, confidence
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg'}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"📁 Saved file: {filepath}")

        # Predict
        pred_class, confidence = predict_image(filepath)
        print(f"🎯 Prediction: {pred_class} ({confidence*100:.2f}%)")

        # Save to DB
        with get_db() as conn:
            conn.execute(
                "INSERT INTO predictions (filename, predicted_class, confidence) VALUES (?, ?, ?)",
                (filename, pred_class, confidence)
            )
            conn.commit()

        return jsonify({
            'class': pred_class,
            'confidence': f"{confidence * 100:.2f}%",
            'image_url': f"/static/uploads/{filename}"
        })

    except Exception as e:
        print(f"🔥 Server error: {str(e)}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)