from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the pickled model and scaler
with open("model.h5", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data.get("scaler") 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order
        features = np.array([[
            data['radius'],
            data['texture'],
            data['perimeter'],
            data['area'],
            data['smoothness'],
            data['compactness'],
            data['concavity'],
            data['concave_points'],
            data['symmetry'],
            data['fractal']
        ]])
        
        # Scale features if scaler is available
        if scaler is not None:
            features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)
        
        # Handle binary prediction (adjust depending on your model)
        probability = float(prediction[0]) if prediction.ndim == 1 else float(prediction[0][0])
        malignant = probability > 0.5
        
        return jsonify({
            'malignant': bool(malignant),
            'probability': probability if malignant else 1 - probability,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
