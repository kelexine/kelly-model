import numpy as np
from sklearn.metrics import accuracy_score
from logger_config import logger

def compute_metrics(pred):
    """
    Computes accuracy metrics.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    logger.info("Evaluation accuracy: %.4f", acc)
    return {'accuracy': acc}

def evaluate_model(trainer, eval_dataset):
    """
    Evaluate the model using the provided trainer and evaluation dataset.
    """
    try:
        logger.info("Starting evaluation...")
        results = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info("Evaluation results: %s", results)
        return results
    except Exception as e:
        logger.error("Error during evaluation", exc_info=True)
        raise e

if __name__ == '__main__':
    logger.debug("Evaluation module executed directly.")