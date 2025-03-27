# flask_api.py
"""
Flask API to interface with Kelly AI.
Provides endpoints for:
  - Loading datasets
  - Predictions (single and batch)
  - Updating the model with corrections
  - Uploading/retrieving custom datasets
  - Retrieving offline knowledge (via FAISS)
  - Performing Google searches
Uses flask-limiter for rate limiting.
"""

import os
from functools import wraps
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from logger_config import setup_logger
from model import Kelly
import database
import requests
import status
import random
import math

app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per hour"])
limiter.init_app(app)
logger = setup_logger("flask_api", "./logs/flask_api.log")

kelly = Kelly()
logger.info("Kelly model initialized for Flask API.")

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or token != os.environ.get("API_SECRET", "default-secret-token"):
            logger.warning("Unauthorized access attempt.")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

def check_training_status(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if status.TRAINING_IN_PROGRESS:
            if status.ESTIMATED_TIME_REMAINING is not None and status.ESTIMATED_TIME_REMAINING > 0:
                minutes = math.floor(status.ESTIMATED_TIME_REMAINING / 60)
                seconds = math.floor(status.ESTIMATED_TIME_REMAINING % 60)
                time_estimate = f"{minutes}m {seconds}s"
            else:
                time_estimate = "unknown"
            humorous_quotes = [
                "Hold tight, I'm busy learning my ABCs!",
                "Training in progress - even AI needs a coffee break!",
                "My neurons are on fire (in a good way)!",
                "I'm polishing my circuits; check back soon!",
                "Currently updating my brain—please wait a moment!"
            ]
            joke = random.choice(humorous_quotes)
            return jsonify({"message": f"Model is busy training. Estimated completion: ~{time_estimate}. {joke}"}), 503
        return f(*args, **kwargs)
    return wrapper

@app.route("/load_dataset", methods=["GET"])
@requires_auth
@check_training_status
@limiter.limit("10 per minute")
def load_dataset_endpoint():
    dataset_name = request.args.get("dataset", "ag_news")
    logger.info(f"API call: load_dataset for {dataset_name}")
    try:
        from datasets import load_dataset
        train_dataset = load_dataset(dataset_name)["train"]
        val_dataset = load_dataset(dataset_name)["test"]
        return jsonify({
            "message": f"Dataset '{dataset_name}' loaded successfully.",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset)
        }), 200
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/load_hf_dataset", methods=["POST"])
@requires_auth
@check_training_status
@limiter.limit("5 per minute")
def load_hf_dataset():
    try:
        data = request.json
        dataset_url = data.get("dataset_url")
        split = data.get("split", "train")
        if not dataset_url:
            return jsonify({"error": "Dataset URL is required"}), 400
        parts = dataset_url.strip("/").split("/")
        if len(parts) < 3 or "datasets" not in parts:
            return jsonify({"error": "Invalid dataset URL format"}), 400
        idx = parts.index("datasets")
        if idx + 2 >= len(parts):
            return jsonify({"error": "Invalid dataset URL format"}), 400
        dataset_identifier = parts[idx + 1] + "/" + parts[idx + 2]
        logger.info(f"Extracted dataset identifier: {dataset_identifier}")
        from datasets import load_dataset
        dataset = load_dataset(dataset_identifier, split=split)
        num_rows = len(dataset)
        columns = dataset.column_names if hasattr(dataset, "column_names") else "Unknown"
        return jsonify({
            "message": f"Dataset '{dataset_identifier}' loaded successfully on split '{split}'.",
            "num_rows": num_rows,
            "columns": columns
        }), 200
    except Exception as e:
        logger.error(f"Error loading dataset from URL: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
@requires_auth
@check_training_status
@limiter.limit("20 per minute")
def predict():
    data = request.json
    task = data.get("task")
    text = data.get("text")
    logger.info(f"API call: predict task {task}")
    try:
        if task == "sentiment":
            result = kelly.analyze_sentiment(text)
        elif task == "summarize":
            result = kelly.summarize_text(text)
        elif task == "generate":
            result = kelly.generate_text(text)
        elif task == "classify":
            result = kelly.classify_text(text)
        elif task == "code_complete":
            result = kelly.code_completion(text)
        elif task == "debug_code":
            result = kelly.debug_code(text)
        elif task == "qa":
            question = data.get("question")
            context = data.get("context")
            result = kelly.answer_question(question, context)
        elif task == "sophisticated_question":
            result = kelly.answer_sophisticated_question(text)
        else:
            result = "Task not recognized."
        from advanced_offline_retrieval import retrieve_offline_knowledge_faiss
        offline_info = retrieve_offline_knowledge_faiss(text, top_k=3)
        combined_result = {"response": result, "offline_knowledge": offline_info}
        logger.info(f"Prediction result: {result[:50]}...")
        return jsonify(combined_result)
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
@requires_auth
@check_training_status
@limiter.limit("10 per minute")
def batch_predict():
    data = request.json
    task = data.get("task")
    texts = data.get("texts")
    logger.info(f"API call: batch_predict for task {task}")
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Invalid or missing 'texts' field. Must be a list."}), 400
    try:
        results = []
        for t in texts:
            if task == "sentiment":
                results.append(kelly.analyze_sentiment(t))
            elif task == "summarize":
                results.append(kelly.summarize_text(t))
            elif task == "generate":
                results.append(kelly.generate_text(t))
            elif task == "classify":
                results.append(kelly.classify_text(t))
            elif task == "code_complete":
                results.append(kelly.code_completion(t))
            elif task == "debug_code":
                results.append(kelly.debug_code(t))
            elif task == "qa":
                question = t.get("question")
                context = t.get("context")
                results.append(kelly.answer_question(question, context))
            elif task == "sophisticated_question":
                results.append(kelly.answer_sophisticated_question(t))
            else:
                return jsonify({"error": "Task not recognized."}), 400
        logger.info(f"Batch prediction results: {str(results)[:100]}...")
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error in batch_predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/update_model", methods=["POST"])
@requires_auth
@limiter.limit("5 per minute")
def update_model():
    data = request.json
    try:
        input_text = data.get("input_text")
        correction = data.get("correction")
        feedback = data.get("feedback")
        logger.info("API call: update_model")
        result = kelly.update_model_with_correction(input_text, correction, feedback)
        return jsonify({"result": result})
    except Exception as e:
        logger.error(f"Error in update_model endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload_dataset", methods=["POST"])
@requires_auth
@limiter.limit("5 per minute")
def upload_dataset():
    data = request.json
    try:
        dataset_name = data.get("name")
        dataset_data = data.get("data")
        logger.info(f"API call: upload_dataset for {dataset_name}")
        if not dataset_name or not dataset_data:
            return jsonify({"error": "Dataset name and data are required."}), 400
        content = f"Dataset {dataset_name}: {dataset_data}"
        database.insert_knowledge(category="dataset", source="admin", content=content)
        logger.info(f"Custom dataset '{dataset_name}' inserted into knowledge base.")
        return jsonify({"result": f"Custom dataset '{dataset_name}' uploaded successfully."})
    except Exception as e:
        logger.error(f"Error in upload_dataset endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_knowledge", methods=["GET"])
@requires_auth
@limiter.limit("20 per minute")
def get_knowledge():
    logger.info("API call: get_knowledge")
    try:
        knowledge_records = database.get_relevant_knowledge("")
        records = [{
            "id": record.id,
            "category": record.category,
            "source": record.source,
            "content": record.content,
            "timestamp": record.timestamp.isoformat()
        } for record in knowledge_records]
        return jsonify({"knowledge": records})
    except Exception as e:
        logger.error(f"Error in get_knowledge endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/retrieve_knowledge", methods=["POST"])
@requires_auth
@limiter.limit("10 per minute")
def retrieve_knowledge():
    data = request.json
    try:
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query is required."}), 400
        from advanced_offline_retrieval import retrieve_offline_knowledge_faiss
        results = retrieve_offline_knowledge_faiss(query, top_k=5)
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error in retrieve_knowledge endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
@requires_auth
@check_training_status
@limiter.limit("10 per minute")
def search_api():
    data = request.json
    try:
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query is required."}), 400
        summary = kelly.search_internet(query)
        return jsonify({"summary": summary})
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask API on port 5000")
    try:
        app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Flask API failed to start: {e}")