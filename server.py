from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

server = Flask(__name__)

# Load trained model package
model_path = "cancer_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Run train_classifier.py first.")

with open(model_path, "rb") as file:
    model_package = pickle.load(file)
    classifier = model_package["classifier"]
    normalizer = model_package["normalizer"]

@server.route('/')
def index():
    return render_template('index.html')

@server.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        input_data = request.get_json()
        
        # Build feature array from input
        feature_vector = np.array([[
            input_data['radius'],
            input_data['texture'],
            input_data['perimeter'],
            input_data['area'],
            input_data['smoothness'],
            input_data['compactness'],
            input_data['concavity'],
            input_data['concave_points'],
            input_data['symmetry'],
            input_data['fractal']
        ]])
        
        # Normalize the features
        normalized_features = normalizer.transform(feature_vector)

        # Get prediction and probability
        prediction_class = classifier.predict(normalized_features)[0]
        prediction_proba = classifier.predict_proba(normalized_features)[0]
        
        # Class 0 = malignant, 1 = benign in sklearn's breast cancer dataset
        is_malignant = (prediction_class == 0)
        confidence = prediction_proba[0] if is_malignant else prediction_proba[1]
        
        return jsonify({
            'malignant': bool(is_malignant),
            'probability': float(confidence),
            'status': 'success'
        })
    
    except KeyError as ke:
        return jsonify({
            'error': f'Missing required field: {str(ke)}',
            'status': 'error'
        }), 400
    except Exception as ex:
        return jsonify({
            'error': f'Prediction failed: {str(ex)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=5000)