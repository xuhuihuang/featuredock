"""
This script is adapted from Bjarten's early-stopping-pytorch project.
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, delta=1e-3, path='best_params.torch'):
        """
        Params
        ------
        patience : int 
            steps/ epochs to wait after last time validation loss improved
        delta : float
            minimum change in the monitored quantity to qualify as an improvement
        path : str 
            path to checkpoint params
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.scores = []
    def __call__(self, val_loss, model):
        score = -val_loss
        self.scores.append(score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        print(f'The best model so far is epoch {len(self.scores)}. Saving the model.')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss