# ewc_trainer.py
from transformers import Trainer
import torch

class EWCTrainer(Trainer):
    def __init__(self, ewc, lambda_ewc: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc = ewc
        self.lambda_ewc = lambda_ewc

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_output = super().compute_loss(model, inputs, return_outputs=True)
        base_loss, outputs = loss_output["loss"], loss_output
        ewc_loss = self.ewc.penalty(model)
        total_loss = base_loss + self.lambda_ewc * ewc_loss
        return (total_loss, outputs) if return_outputs else total_loss