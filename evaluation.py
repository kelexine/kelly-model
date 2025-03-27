# evaluation.py
"""
Evaluation module for Kelly AI.
Uses Hugging Face Trainer to evaluate the model.
"""
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from logger_config import setup_logger
from config import Config

logger = setup_logger("evaluation", "./logs/evaluation.log")

def evaluate_model(kelly, val_dataset):
    logger.info("Starting evaluation of Kelly's model.")
    try:
        model = kelly.classifier_model
        data_collator = DataCollatorWithPadding(tokenizer=kelly.tokenizer)
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=Config.Training.BATCH_SIZE,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=val_dataset,
            tokenizer=kelly.tokenizer,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()
        logger.info(f"Evaluation complete. Metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}