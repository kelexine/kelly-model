from flask import Flask, request, jsonify
import threading
import time
import requests
from logger_config import logger
from database import get_db_connection

# Basic in-memory rate limiter: {ip: [timestamp, count]}
RATE_LIMIT = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_COUNT = 10   # max requests per window

# Simple admin token (in production, use secure authentication)
ADMIN_TOKEN = "supersecrettoken"

app = Flask(__name__)

def rate_limiter(ip):
    from time import time
    current_time = time()
    window = RATE_LIMIT.get(ip, {"start": current_time, "count": 0})
    if current_time - window["start"] > RATE_LIMIT_WINDOW:
        RATE_LIMIT[ip] = {"start": current_time, "count": 1}
        return True
    elif window["count"] < RATE_LIMIT_COUNT:
        window["count"] += 1
        RATE_LIMIT[ip] = window
        return True
    else:
        return False

@app.before_request
def limit_remote_addr():
    ip = request.remote_addr
    if not rate_limiter(ip):
        return jsonify({"error": "Too many requests"}), 429

@app.route('/admin/load_dataset', methods=['GET'])
def load_dataset():
    token = request.args.get('token', '')
    if token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        logger.info("Admin requested dataset load from URL: %s", url)
        response = requests.get(url)
        response.raise_for_status()
        # For simplicity, assume dataset is returned in JSON format
        dataset = response.json()
        # In a production system, we might save this dataset to a database or file system
        # Here, we simulate saving to the database
        conn = get_db_connection()
        conn.execute("INSERT INTO datasets (url, data) VALUES (?, ?)", (url, str(dataset)))
        conn.commit()
        conn.close()
        return jsonify({"status": "Dataset loaded successfully", "data": dataset})
    except Exception as e:
        logger.error("Error loading dataset", exc_info=True)
        return jsonify({"error": "Failed to load dataset", "details": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if not json_data or "text" not in json_data:
            return jsonify({"error": "No text provided for prediction"}), 400
        text = json_data["text"]
        # For production, integrate the actual model inference here.
        # For demo, we return a dummy classification.
        result = {"text": text, "prediction": "positive", "confidence": 0.95}
        logger.info("Prediction made for input: %s", text)
        return jsonify(result)
    except Exception as e:
        logger.error("Error during prediction", exc_info=True)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

def run_api():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    logger.info("Starting Flask API...")
    run_api()