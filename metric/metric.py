import torch

def eval_accuracy(pred, label):
    if len(pred.shape) > 1:
        return torch.sum(torch.argmax(pred, dim=-1) == label).item() / len(label) 
    else:
        return torch.sum((pred > 0.5) == label).item() / len(label)

def eval_mse(pred, label):
    return torch.mean((pred - label) ** 2).item()

def eval_mae(pred, label):
    return torch.mean(torch.abs(pred - label)).item()

def eval_rmse(pred, label):
    return torch.sqrt(torch.mean((pred - label) ** 2)).item()


optional_metric = {
    "acc": eval_accuracy,
    "mse": eval_mse,
    "mae": eval_mae,
    "rmse": eval_rmse,
    
}

"""
    metric in greater_metric should be greater is better
"""
greater_metric = {
    "acc", "f1", "auc"
}

"""
    metric in lower_metric should be lower is better
"""
lower_metric = {
    "loss", "mse"
}


def eval_all_metric(config, pred, label):
    """
        eval model on all metric in config.eval_metric
    """
    result_dict = {}
    result_dict["data_size"] = len(label)
    for metric in config.eval_metric:
        if metric in optional_metric:
            result_dict[metric] = optional_metric[metric](pred, label)
        else:
            raise NotImplementedError(f"metric {metric} not implemented")
    return result_dict

def merge_result(result_dict1, result_dict2):
    """
        merge two result dict use average strategy
    """

    # if one of the result dict is None, return the other one
    if result_dict1 is None:
        return result_dict2
    if result_dict2 is None:
        return result_dict1
    
    # else merge two result dict
    result_dict = {}
    for metric in result_dict1:
        if metric == "data_size":
            continue
        part1 = result_dict1[metric] * result_dict1["data_size"]
        part2 = result_dict2[metric] * result_dict2["data_size"]
        result_dict[metric] = (part1 + part2) / (result_dict1["data_size"] + result_dict2["data_size"])
    result_dict["data_size"] = result_dict1["data_size"] + result_dict2["data_size"]
    return result_dict

def eval_reduce(result_list):
    reduced_result = None
    for result in result_list:
        reduced_result = merge_result(reduced_result, result)
    return reduced_result

def eval_compare(metric, cur, best):
    if best is None:
        return True
    if metric in greater_metric:
        return cur > best
    else:
        return cur < best