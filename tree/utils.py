"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import numpy as np
import pandas as pd


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # Check if all elements are numeric
    if pd.api.types.is_numeric_dtype(y):
        # Check if all values are integers
        if y.dtype == int or y.dtype == float:
            return True  # All values are real (continuous)
        else:
            return False  # Values are discrete
    else:
        return False  # Values are not numeric

def entropy(y: pd.Series) -> float:
    """
    Function to calculate the entropy of a given series
    """
    # Count the occurrences of each unique value in the Series
    value_counts = y.value_counts()

    # Calculate the probability of each unique value
    probabilities = value_counts / len(y)

    # Calculate entropy using the formula: -sum(p * log2(p))
    entropy_value = -sum(probabilities * np.log2(probabilities))

    return entropy_value


def gini_index(y: pd.Series) -> float:
    """
    Function to calculate the Gini index of a given series
    """
    # Count the occurrences of each unique value in the Series
    value_counts = y.value_counts()

    # Calculate the probability of each unique value
    probabilities = value_counts / len(y)

    # Calculate Gini index using the formula: 1 - sum(p^2)
    gini_index_value = 1 - sum(probabilities**2)

    return gini_index_value



def information_gain(y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain of a given attribute in a dataset
    """

    # Calculate the entropy before splitting (entropy of the original dataset)
    entropy_before = entropy(y)

    # Combine the target variable and the attribute for easier calculations
    combined_data = pd.DataFrame({'target': y, 'attribute': attr})

    # Calculate the weighted average entropy after splitting
    entropy_after = combined_data.groupby('attribute')['target'].apply(lambda x: entropy(x) * len(x) / len(y)).sum()

    # Calculate information gain using the formula: entropy_before - entropy_after
    information_gain_value = entropy_before - entropy_after

    return information_gain_value


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split upon.
    If needed you can split this function into 2, one for discrete and one for real-valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # Initialize variables to store the best attribute and its corresponding information gain or Gini index
    best_attribute = None
    best_score = float('-inf') if criterion in ['entropy', 'information_gain'] else float('inf')

    # Iterate over each feature and calculate information gain or Gini index
    for attribute in features:
        if criterion in ['entropy', 'information_gain']:
            # Calculate information gain for real-valued features
            info_gain = information_gain(y, X[attribute])
            if info_gain > best_score:
                best_score = info_gain
                best_attribute = attribute
        elif criterion == 'gini':
            # Calculate Gini index for discrete-valued features
            gini_idx = gini_index(X[attribute])
            if gini_idx < best_score:
                best_score = gini_idx
                best_attribute = attribute

    return best_attribute


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Function to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real-valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data (Input and output)
    """
    # Check if the attribute is real or discrete-valued
    if X[attribute].dtype == int or X[attribute].dtype == float:
        # For real-valued attributes
        left_data = X[X[attribute] <= value]
        right_data = X[X[attribute] > value]
    else:
        # For discrete-valued attributes
        left_data = X[X[attribute] == value]
        right_data = X[X[attribute] != value]

    # Split the target variable accordingly
    left_target = y.loc[left_data.index]
    right_target = y.loc[right_data.index]

    return left_data, left_target, right_data, right_target