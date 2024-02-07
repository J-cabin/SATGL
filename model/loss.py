import torch
import torch.nn as nn
import torch.nn.functional as fn

def cross_entropy(pred, label):
    return fn.cross_entropy(pred, label.to(torch.long))

def binary_cross_entropy(pred, label):
    return nn.BCELoss()

optional_loss_func = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "binary_cross_entropy": nn.BCELoss(),
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss()
}

def get_loss(config):
    loss_name = config.model_settings["loss"]
    if loss_name in optional_loss_func:
        return optional_loss_func[loss_name]
    else:
        raise NotImplementedError(f"loss function {loss_name} not implemented")