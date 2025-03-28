from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from logger_config import logger

def build_model(model_name="distilbert-base-uncased", num_labels=2, model_path=None):
    """
    Build and return a DistilBERT-based model for sequence classification.
    If a model_path is provided and exists, load the fine-tuned model from disk.
    Otherwise, load the base DistilBERT model.
    """
    try:
        logger.info("Loading tokenizer and model...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        if model_path is not None:
            # Attempt to load the fine-tuned model from the given path
            logger.info("Loading fine-tuned model from %s", model_path)
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            # Load the base model
            model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        logger.info("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error("Error building model", exc_info=True)
        raise e

if __name__ == '__main__':
    model, tokenizer = build_model()
    logger.debug("Model and tokenizer ready for training/inference.")