import os
import threading
import time
from logger_config import logger
from model import build_model
from training import train_model
from preprocessing import Preprocessor
from flask_api import run_api
from database import initialize_database

FINETUNED_MODEL_PATH = "./kelly_finetuned"

def interactive_mode(model, tokenizer):
    logger.info("Entering interactive mode. Type 'exit' to quit.")
    # Interactive loop: perform inference using the loaded model.
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            logger.info("Exiting interactive mode.")
            break
        try:
            # Tokenize input and perform inference
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            # For classification, get predicted label (argmax) and dummy confidence
            pred_label = outputs.logits.argmax(dim=1).item()
            # In a production system, you'd map label indices to actual labels
            logger.info("Inference result for '%s': Label %s", user_input, pred_label)
            print(f"Input: {user_input}\nPredicted Label: {pred_label}")
        except Exception as e:
            logger.error("Error during inference", exc_info=True)
            print("An error occurred during inference.")

def training_mode():
    logger.info("Starting training mode.")
    # Initialize the database
    initialize_database()
    # Placeholder: Load a sample dataset. In production, replace with actual data loading.
    import pandas as pd
    data = {'text': ["This is great", "This is terrible"], 'label': [1, 0]}
    dataset = pd.DataFrame(data)
    preprocessor = Preprocessor()
    analysis = preprocessor.analyze_dataset(dataset)
    logger.info("Dataset analysis for training: %s", analysis)
    # Build model and tokenizer (base model)
    model, tokenizer = build_model()
    # Create a dummy torch Dataset for demonstration purposes.
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    dummy_dataset = DummyDataset(dataset['text'].tolist(), dataset['label'].tolist(), tokenizer)
    # Train model (fine-tuning)
    train_model(model, tokenizer, dummy_dataset, output_dir=FINETUNED_MODEL_PATH)
    logger.info("Training mode completed. Fine-tuned model saved to %s", FINETUNED_MODEL_PATH)
    return model, tokenizer

def start_flask_api_after_delay(delay=20):
    def delayed_start():
        logger.info("Waiting %s seconds before starting Flask API...", delay)
        time.sleep(delay)
        logger.info("Starting Flask API now.")
        run_api()
    thread = threading.Thread(target=delayed_start, daemon=True)
    thread.start()

def main():
    # Check if a fine-tuned model exists
    if os.path.exists(FINETUNED_MODEL_PATH) and os.listdir(FINETUNED_MODEL_PATH):
        logger.info("Fine-tuned model detected. Loading model for interactive inference.")
        model, tokenizer = build_model(model_path=FINETUNED_MODEL_PATH)
        start_flask_api_after_delay()
        interactive_mode(model, tokenizer)
    else:
        logger.info("No fine-tuned model found. Launching training mode.")
        model, tokenizer = training_mode()
        # After training, load the fine-tuned model for interactive inference.
        model, tokenizer = build_model(model_path=FINETUNED_MODEL_PATH)
        start_flask_api_after_delay()
        interactive_mode(model, tokenizer)

if __name__ == '__main__':
    main()