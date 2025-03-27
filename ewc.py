# ewc.py
"""
Elastic Weight Consolidation (EWC) module for Kelly AI.
Computes the Fisher Information matrix on a given dataset and stores a snapshot of model parameters.
This is used to add a regularization penalty during continual learning.
"""
import torch

class EWC:
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.fisher = self.compute_fisher()
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def compute_fisher(self):
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)
        self.model.train()
        for batch in self.dataloader:
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs["loss"]
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(self.dataloader)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss