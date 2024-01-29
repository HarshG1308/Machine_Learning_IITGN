from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size, "Input sizes must be equal."
    return (y_hat == y).mean()

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "Input sizes must be equal."
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_positive = ((y_hat == cls) & (y != cls)).sum()
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "Input sizes must be equal."
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_negative = ((y_hat != cls) & (y == cls)).sum()
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (RMSE)
    """
    assert y_hat.size == y.size, "Input sizes must be equal."
    return np.sqrt(((y_hat - y) ** 2).mean())

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (MAE)
    """
    assert y_hat.size == y.size, "Input sizes must be equal."
    return np.abs(y_hat - y).mean()
