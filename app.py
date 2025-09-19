import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# --- Load the exported machine learning artifacts ---
try:
    rf_model = joblib.load('rf_model.joblib')
    ann_model = load_model('ann_model.h5')
    preprocessor = joblib.load('preprocessor.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Models and preprocessors loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    rf_model = ann_model = preprocessor = scaler = None

# --- Re-implement the Logistic Regression model from the notebook ---
lr_weights = np.array([-0.14, -0.21, -0.04, -0.01, -0.09, 0.04, -0.12, 0.03, -0.07, 0.13, 
                       0.19, 0.03, 0.01, 0.08, 0.02, 0.01, 0.04, 0.03, 0.02, 0.01, 
                       0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                       0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                       0.01, 0.01, 0.01, 0.01, 0.01, 0, -1.02, 0.45, 0.98, 0.15, 
                       0.1, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

lr_bias = 0.02

X_train_mean = np.array([0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 
                        0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 
                        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                        0.05, 0.05, 0, 92.2, 59.6, 7.5, 172.5, 8.1, 10.3, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0])

X_train_std = np.array([0.3, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 
                       0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 
                       0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 
                       0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 50.5, 33.3, 
                       2.3, 30.5, 2.9, 4.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def predict_logistic_regression(scaled_vector):
    """Manual logistic regression prediction"""
    safe_std = np.where(X_train_std == 0, 1, X_train_std)
    manual_scaled_vector = (scaled_vector - X_train_mean) / safe_std
    z = np.dot(manual_scaled_vector, lr_weights) + lr_bias
    return sigmoid(z)

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not all([rf_model, ann_model, preprocessor, scaler]):
        return jsonify({'error': 'Models not loaded. Server is not ready.'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Create DataFrame with input data
        input_df = pd.DataFrame([data], columns=[
            'batting_team', 'bowling_team', 'city', 'runs_left', 
            'balls_left', 'wickets_left', 'runs_target', 'crr', 'rrr'
        ])
        
        # Process the data
        processed_vector = preprocessor.transform(input_df)
        scaled_vector_ann = scaler.transform(processed_vector)
        
        # Get predictions from all models
        rf_prob = rf_model.predict_proba(processed_vector)[0][1]
        ann_prob = ann_model.predict(scaled_vector_ann)[0][0]
        lr_prob = predict_logistic_regression(processed_vector[0])
        
        # Calculate weighted average (ensemble prediction)
        final_prob = (rf_prob * 0.45) + (ann_prob * 0.45) + (lr_prob * 0.10)
        
        # Prepare response
        response = {
            'final_prediction': float(final_prob),
            'model_breakdown': {
                'random_forest': float(rf_prob),
                'ann': float(ann_prob),
                'logistic_regression': float(lr_prob)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Please check your input data.'}), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# --- Run the server ---
if __name__ == '__main__':
    # This runs the server in debug mode on port 5000 for local development.
    print("Starting IPL Win Predictor Server...")
    print("Server will be available at: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)