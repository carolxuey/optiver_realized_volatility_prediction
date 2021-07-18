import os
import random
import numpy as np
import torch


def set_seed(seed, deterministic_cudnn=False):

    """
    Set random seed for reproducible results

    Parameters
    ----------
    seed (int): Random seed
    deterministic_cudnn (bool): Whether to set deterministic cuDNN or not
    """

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmspe_metric(y_true, y_pred):

    """
    Calculate root mean squared percentage error between ground-truth and predictions

    Parameters
    ----------
    y_true [array-like of shape (n_samples)]: Ground-truth
    y_pred [array-like of shape (n_samples)]: Predictions

    Returns
    -------
    rmspe (float): Root mean squared percentage error
    """

    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return rmspe


def rmspe_loss_pt(y_true, y_pred):

    """
    Calculate root mean squared percentage error between ground-truth and predictions

    Parameters
    ----------
    y_true [torch.tensor of shape (n_samples)]: Ground-truth
    y_pred [torch.tensor of shape (n_samples)]: Predictions

    Returns
    -------
    rmspe (torch.FloatTensor): Root mean squared percentage error
    """

    rmspe = torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true)))
    return rmspe


def rmspe_eval_lgb(y_pred, train_dataset):

    y_true = train_dataset.get_label()
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return 'rmspe', rmspe, False


def rmspe_loss_lgb(y_pred, train_dataset):

    y_true = train_dataset.get_label()
    grad = 2.0 / y_true * (y_pred * 1.0 / y_true - 1)
    hess = 2.0 / (y_true ** 2)
    return grad, hess
