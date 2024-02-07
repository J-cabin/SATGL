import torch.optim as optim

optional_optimizer = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "rmsprop": optim.RMSprop
}

def get_optimizer(config):
    if config.optimizer in optional_optimizer:
        return optional_optimizer[config.optimizer]
    else:
        raise NotImplementedError(f"optimizer {config.optimizer} not implemented")