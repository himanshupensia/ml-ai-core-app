from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model at startup
model_path = os.environ.get('MODEL_PATH', 'model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/v1/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Expecting JSON input: {"data": [[5.1, 3.5, 1.4, 0.2]]}
        json_input = request.get_json()
        input_data = np.array(json_input['data'])
        
        prediction = model.predict(input_data)
        
        # Return prediction as list
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # AI Core typically expects traffic on a specific port, e.g., 9001
    app.run(host='0.0.0.0', port=9001)