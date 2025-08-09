from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Initialize model as None - will be loaded when needed
model = None
model_info = None

def load_model():
    """Load the best model from pickle file"""
    global model, model_info
    if model is None:
        try:
            # Load the model from pickle file
            with open('models/best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Load model metadata
            with open('models/model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
                
            print(f"Loaded {model_info['model_name']} model with RMSE: {model_info['rmse']}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
            model_info = None
    return model

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/info', methods=['GET'])
def info():
    """Model info endpoint"""
    try:
        current_model = load_model()
        if current_model is not None and model_info is not None:
            return jsonify({
                'model_loaded': True,
                'model_name': model_info['model_name'],
                'rmse': model_info['rmse'],
                'feature_count': len(model_info['feature_names'])
            })
        else:
            return jsonify({
                'model_loaded': False,
                'error': 'Model not available'
            })
    except Exception as e:
        return jsonify({
            'model_loaded': False,
            'error': str(e)
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Prediction request received")
        
        # Load model if not already loaded
        current_model = load_model()
        if current_model is None:
            print("Model is None - not loaded")
            return jsonify({'error': 'Model not available'}), 500
        
        print("Model loaded successfully")
        
        # Get JSON data from request
        data = request.get_json(force=True)
        print(f"Received data: {data}")
        
        # Convert to DataFrame if it's a list of features
        if isinstance(data, list):
            # Assuming the input is a list of feature values
            df = pd.DataFrame([data])
        elif isinstance(data, dict):
            # If it's a dictionary, convert to DataFrame
            if 'features' in data:
                df = pd.DataFrame([data['features']])
            else:
                df = pd.DataFrame([data])
        else:
            return jsonify({'error': 'Invalid input format'}), 400
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Make prediction
        prediction = current_model.predict(df)
        print(f"Prediction: {prediction}")
        
        # Return prediction
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)