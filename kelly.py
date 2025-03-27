# kelly.py
"""
Main driver script for Kelly AI.
On first run, the model is sequentially fine-tuned on multiple datasets using dynamic, task-aware preprocessing:
  - AG News (classification, "text")
  - English Sentiment Analysis (classification, "text")
  - English Dictionary (combining "word" and "definition" into "text")
  - English Jokes (humor, "joke")
After sequential training, the model is saved.
A continuous learning loop runs in a background thread to rehearse new corrections (using EWC and a replay buffer),
marking them as rehearsed to avoid over-fitting.
During interactive sessions, offline retrieval (via FAISS) supplements model responses.
The Flask API is deployed with rate limiting. For production, deploy using Gunicorn.
"""
import os
import time
import threading
from datasets import Value
import advanced_preprocessing as ap
import model
import training
import continuous_learning as cl
from logger_config import setup_logger
from config import Config
from dependency_injection import create_kelly

logger = setup_logger("kelly", "./logs/kelly.log")
TRAINED_MODEL_DIR = Config.MODEL_DIR

def interactive_mode(kelly_instance):
    logger.info("Entering interactive mode. Type your prompt (or 'exit' to quit):")
    from advanced_offline_retrieval import retrieve_offline_knowledge_faiss
    while True:
        try:
            user_input = input("Prompt: ")
            if user_input.lower() == "exit":
                break
            answer = kelly_instance.answer_sophisticated_question(user_input)
            offline_info = retrieve_offline_knowledge_faiss(user_input, top_k=3)
            print("Response:", answer)
            if offline_info:
                print("\nOffline Knowledge:")
                for item in offline_info:
                    print(f"- {item['content']} (Similarity: {item['similarity']:.2f})")
        except KeyboardInterrupt:
            break

def start_flask_api():
    import flask_api  # flask_api.py will call app.run()

def sequential_training(model_instance, datasets_info):
    for ds in datasets_info:
        ds_id = ds["id"]
        config = ds.get("config", {})
        logger.info(f"Processing dataset {ds_id} with config {config}")
        try:
            train_ds, val_ds = ap.preprocess_dataset_dynamic(ds_id, config=config, max_length=128)
            if "label" in train_ds.column_names:
                train_ds = train_ds.cast_column("label", Value("int64"))
            if "label" in val_ds.column_names:
                val_ds = val_ds.cast_column("label", Value("int64"))
            logger.info(f"Fine-tuning model on dataset {ds_id}...")
            training.fine_tune_model(model_instance, train_ds, val_ds)
            logger.info(f"Finished training on dataset {ds_id}. Model updated and saved.")
        except Exception as e:
            logger.error(f"Error during training on dataset {ds_id}: {e}")

def continuous_learning_loop(model_instance, tokenizer, interval_seconds=Config.ContinuousLearning.INTERVAL_SECONDS):
    while True:
        try:
            logger.info("Starting continuous learning cycle...")
            cl.rehearsal_finetune(model_instance, tokenizer, num_epochs=Config.ContinuousLearning.NUM_EPOCHS, batch_size=Config.ContinuousLearning.BATCH_SIZE)
            logger.info("Continuous learning cycle complete. Sleeping...")
        except Exception as e:
            logger.error(f"Error in continuous learning cycle: {e}")
        time.sleep(interval_seconds)

def main():
    logger.info("Starting Kelly AI project (adaptive multi-task training)...")
    model_exists = os.path.exists(TRAINED_MODEL_DIR) and os.path.exists(os.path.join(TRAINED_MODEL_DIR, "tokenizer_config.json"))
    kelly_instance = None

    if not model_exists:
        logger.info("No fine-tuned model found. Starting sequential training on default datasets...")
        kelly_instance = create_kelly()
        datasets_info = [
            {"id": "ag_news", "config": {"task": "classification", "input_fields": ["text"]}},
            {"id": "Juanid14317/EnglishSentimentAnalysis", "config": {"task": "classification", "input_fields": ["text"]}},
            {"id": "MAKILINGDING/english_dictionary", "config": {"task": "classification", "input_fields": ["word", "definition"]}},
            {"id": "kuldin/english_jokes", "config": {"task": "humor", "input_fields": ["joke"]}}
        ]
        sequential_training(kelly_instance, datasets_info)
        logger.info("Sequential training complete. Model saved in './kelly_finetuned'")
    else:
        logger.info("Fine-tuned model found. Loading model for interactive mode...")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)
            classifier_model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_DIR)
            kelly_instance = create_kelly(tokenizer_name=TRAINED_MODEL_DIR, classifier_model_name=TRAINED_MODEL_DIR)
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            return

    if kelly_instance is not None:
        cl_thread = threading.Thread(target=continuous_learning_loop, args=(kelly_instance, kelly_instance.tokenizer), daemon=True)
        cl_thread.start()

    interactive_mode(kelly_instance)

if __name__ == "__main__":
    main()