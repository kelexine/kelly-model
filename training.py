# training.py
"""
Training module for Kelly AI.
Uses Hugging Face Trainer with DeepSpeed for distributed training and integrates MLflow for experiment tracking.
Supports advanced continual learning with EWC via a custom EWCTrainer.
"""
import evaluate
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from logger_config import setup_logger
import status
import time
from config import Config
import mlflow

logger = setup_logger("training", "./logs/training.log")

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if state.max_steps > 0 and state.global_step > 0:
            progress = state.global_step / state.max_steps
            elapsed = time.time() - status.TRAINING_START_TIME
            estimated_total = elapsed / progress
            status.ESTIMATED_TIME_REMAINING = estimated_total - elapsed

def compute_metrics(eval_pred):
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    acc = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

def fine_tune_model(kelly, train_dataset, val_dataset, use_ewc=False, ewc_instance=None, lambda_ewc=Config.ContinuousLearning.EWC_LAMBDA):
    logger.info("Starting fine-tuning process.")
    mlflow.start_run()
    mlflow.log_params({
        "epochs": Config.Training.EPOCHS,
        "batch_size": Config.Training.BATCH_SIZE,
        "learning_rate": Config.Training.LEARNING_RATE
    })
    try:
        status.TRAINING_IN_PROGRESS = True
        status.TRAINING_START_TIME = time.time()
        model_to_finetune = kelly.classifier_model
        data_collator = DataCollatorWithPadding(tokenizer=kelly.tokenizer)
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=Config.Training.EPOCHS,
            per_device_train_batch_size=Config.Training.BATCH_SIZE,
            per_device_eval_batch_size=Config.Training.BATCH_SIZE,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=Config.Training.LEARNING_RATE,
            weight_decay=0.01,
            logging_dir="./logs/trainer",
            logging_steps=Config.Training.LOGGING_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            deepspeed=Config.Training.DEEPSPEED_CONFIG if Config.Training.USE_DEEPSPEED else None,
        )
        if use_ewc and ewc_instance is not None:
            from ewc_trainer import EWCTrainer
            trainer = EWCTrainer(
                ewc=ewc_instance,
                lambda_ewc=lambda_ewc,
                model=model_to_finetune,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=kelly.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        else:
            trainer = Trainer(
                model=model_to_finetune,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=kelly.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        trainer.add_callback(ProgressCallback)
        trainer.train()
        mlflow.log_metric("final_loss", trainer.state.loss)
        logger.info("Fine-tuning complete. Saving model and tokenizer.")
        model_to_finetune.save_pretrained(Config.MODEL_DIR)
        kelly.tokenizer.save_pretrained(Config.MODEL_DIR)
        logger.info(f"Waiting {Config.API.DELAY_BEFORE_API} seconds before starting the Flask API...")
        threading.Timer(Config.API.DELAY_BEFORE_API, start_flask_api).start()
    except Exception as e:
        logger.error(f"Training error: {e}")
        mlflow.log_metric("error", 1)
        raise e
    finally:
        status.TRAINING_IN_PROGRESS = False
        status.ESTIMATED_TIME_REMAINING = 0
        mlflow.end_run()
    return trainer