"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
    
    def find_best_split(X,y):
    # Placeholder implementation, replace with your logic to find the best split
        best_attribute = X.columns[0]
        threshold = X[best_attribute].mean()
        return best_attribute, threshold

    def create_leaf_node(y):
        # Placeholder implementation, replace with your logic to create a leaf node
        return {'class': y.mode()[0]}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Check if the maximum depth is reached or if all instances belong to the same class
        if depth == self.max_depth or y.nunique() == 1:
            return create_leaf_node(y)

        # Find the best attribute to split on and the corresponding threshold
        best_attribute, threshold = find_best_split(X, y)

        # If no suitable split is found, create a leaf node
        if best_attribute is None:
            return create_leaf_node(y)

        # Split the dataset based on the best attribute and threshold
        left_mask = X[best_attribute] <= threshold
        right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Create a dictionary to represent the current split
        return {'attribute': best_attribute, 'threshold': threshold,
                'left_subtree': left_subtree, 'right_subtree': right_subtree}

# Assume you have a Node class and other necessary functions like find_best_split and create_leaf_node in utils.py

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        if self.tree is None:
            raise ValueError("The decision tree has not been trained.")

        return X.apply(lambda row: self._traverse_tree(row, self.tree), axis=1)

    def _traverse_tree(self, x, node):
        # Recursively traverse the tree to make predictions for a single test input
        if 'class' in node:
            # Leaf node, return the predicted class
            return node['class']
        else:
            # Internal node, determine which subtree to traverse based on the split criteria
            attribute = node['attribute']
            threshold = node['threshold']

            if x[attribute] <= threshold:
                return self._traverse_tree(x, node['left_subtree'])
            else:
                return self._traverse_tree(x, node['right_subtree'])


    def plot(self,X: pd.DataFrame) -> None:
        """
        Function to plot the tree
        """
        if self.tree is None:
            raise ValueError("The decision tree has not been trained.")

        self._plot_tree(self.tree, depth=0, feature_names=X.columns)

    def _plot_tree(self, node, depth, feature_names):
        if 'class' in node:
            # Leaf node
            print(f"{'  ' * depth}Class: {node['class']}")
        else:
            # Internal node
            attribute = node['attribute']
            threshold = node['threshold']
            print(f"{'  ' * depth}?({attribute} > {threshold})")
            print(f"{'  ' * (depth + 1)}Y:", end=" ")
            self._plot_tree(node['left_subtree'], depth + 1, feature_names)
            print(f"{'  ' * (depth + 1)}N:", end=" ")
            self._plot_tree(node['right_subtree'], depth + 1, feature_names)