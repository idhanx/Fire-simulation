import torch.nn as nn

def get_loss():
    return nn.BCEWithLogitsLoss(pos_weight=None)