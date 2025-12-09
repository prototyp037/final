from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='../frontend')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/generate', methods=['POST'])
def generate_music():
    # Placeholder for AI generation logic
    data = request.json
    current_notes = data.get('notes', [])
    
    # TODO: Call PyTorch model here
    
    return jsonify({
        "status": "success",
        "message": "Generation not implemented yet",
        "notes": current_notes # Just echo back for now
    })

if __name__ == '__main__':
    print("Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)

