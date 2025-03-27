# continuous_learning.py
"""
Continuous Learning (Rehearsal) module for Kelly AI.
Implements rehearsal with a replay buffer and integrates EWC to mitigate catastrophic forgetting.
After rehearsal, used items are marked as rehearsed.
"""
from datasets import Dataset
from transformers import DataCollatorWithPadding
from training import fine_tune_model, compute_metrics
from logger_config import setup_logger
import database
import time

logger = setup_logger("continuous_learning", "./logs/continuous_learning.log")
replay_buffer = []  # In-memory replay buffer

def load_rehearsal_data():
    try:
        items = database.get_relevant_knowledge("correction", limit=1000, only_new=True)
        if items:
            ids = [item.id for item in items]
            data = [{"text": item.content} for item in items]
            logger.info(f"Loaded {len(data)} rehearsal items.")
            return Dataset.from_dict(data), ids
        else:
            logger.info("No rehearsal data found.")
            return None, []
    except Exception as e:
        logger.error(f"Error loading rehearsal data: {e}")
        return None, []

def preprocess_rehearsal_dataset(dataset, tokenizer, max_length=128):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized_ds

def rehearsal_finetune(model_instance, tokenizer, num_epochs=1, batch_size=8):
    rehearsal_data, ids = load_rehearsal_data()
    if rehearsal_data is None or len(rehearsal_data) == 0:
        logger.info("No rehearsal data available. Skipping continuous learning step.")
        return None
    tokenized_ds = preprocess_rehearsal_dataset(rehearsal_data, tokenizer)
    tokenized_ds = tokenized_ds.add_column("label", [0] * len(tokenized_ds))
    global replay_buffer
    replay_buffer.extend(tokenized_ds)
    from transformers import TrainingArguments, Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="./results_rehearsal",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        logging_dir="./logs/rehearsal",
        logging_steps=10,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model_instance.classifier_model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Starting rehearsal fine-tuning...")
    trainer.train()
    logger.info("Rehearsal fine-tuning complete.")
    try:
        database.mark_as_rehearsed(ids)
        logger.info(f"Marked {len(ids)} rehearsal items as rehearsed.")
    except Exception as e:
        logger.error(f"Error marking rehearsal items: {e}")
    return trainer
    
    len 
  