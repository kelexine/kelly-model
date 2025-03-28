import os
import torch
from transformers import Trainer, TrainingArguments
from logger_config import logger

def train_model(model, tokenizer, train_dataset, output_dir="./kelly_finetuned", num_train_epochs=3, batch_size=8):
    """
    Fine-tunes the model using Hugging Face Trainer and saves the fine-tuned model weights
    and all associated files (configuration, tokenizer, etc.) needed for subsequent training or inference.
    
    Parameters:
    - model: The pre-loaded model to be fine-tuned.
    - tokenizer: The tokenizer corresponding to the model.
    - train_dataset: A torch Dataset or Hugging Face Dataset containing the training data.
    - output_dir: Directory path where the fine-tuned model and associated files will be saved.
    - num_train_epochs: Total number of training epochs.
    - batch_size: Batch size per device during training.
    """
    try:
        logger.info("Starting training process...")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,                        # Directory for model checkpoints and outputs.
            num_train_epochs=num_train_epochs,            # Total number of training epochs.
            per_device_train_batch_size=batch_size,       # Batch size per device during training.
            evaluation_strategy="no",                     # No evaluation during training in this example.
            save_strategy="epoch",                        # Save checkpoint at the end of each epoch.
            logging_dir='./logs',                         # Directory for storing logs.
            logging_steps=10,                             # Log every 10 steps.
            load_best_model_at_end=True,                  # Load the best model when finished training.
            save_total_limit=2,                           # Limit the total number of saved checkpoints.
        )

        # Initialize the Trainer with the model, tokenizer, and training dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        # Begin the training process
        trainer.train()
        logger.info("Training completed successfully.")

        # Save the fine-tuned model along with its configuration and optimizer state
        logger.info("Saving the fine-tuned model and associated files to %s", output_dir)
        trainer.save_model(output_dir)

        # Also save the tokenizer configuration to the output directory
        tokenizer.save_pretrained(output_dir)
        logger.info("Model and tokenizer saved successfully to %s", output_dir)

    except Exception as e:
        logger.error("Error during training", exc_info=True)
        raise e

if __name__ == '__main__':
    # For direct testing purposes, log a debug message
    logger.debug("Training module executed directly.")