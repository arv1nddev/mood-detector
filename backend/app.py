from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os

app = Flask(__name__, static_folder='..', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load trained model and vectorizer using absolute paths
try:
    model = joblib.load(os.path.join(script_dir, "mood_model.pkl"))
    vectorizer = joblib.load(os.path.join(script_dir, "vectorizer.pkl"))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def send_file(path):
    return send_from_directory('../frontend', path)

@app.route('/mood', methods=['POST'])
def detect_mood():
    try:
        if not model or not vectorizer:
            return jsonify({'error': 'Model not loaded properly'}), 500
            
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text in JSON format'}), 400
            
        user_text = data['text']
        if not user_text:
            return jsonify({'error': 'Text cannot be empty'}), 400
            
        features = vectorizer.transform([user_text])
        prediction = model.predict(features)[0]
        
        return jsonify({
            'mood': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
