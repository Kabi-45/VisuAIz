import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes


class DatasetLoader:
    def __init__(self):
        """Initialize dataset attributes."""
        self.X = None
        self.y = None

    def load_dataset(self, dataset_name):
        """
        Loads the selected dataset and assigns values to X and y.

        Parameters:
        - dataset_name (str): Name of the dataset ("Iris", "Diabetes", or "Custom CSV").

        Returns:
        - X (DataFrame): Features of the dataset.
        - y (Series): Target variable.
        """
        if dataset_name == "Iris":
            data = load_iris()
            self.X = pd.DataFrame(data.data, columns=data.feature_names)
            self.y = pd.Series(data.target)

        elif dataset_name == "Diabetes":
            data = load_diabetes()
            self.X = pd.DataFrame(data.data, columns=data.feature_names)
            self.y = pd.Series(data.target)

        elif dataset_name == "Custom CSV":
            self._load_custom_csv()

        return self.X, self.y

    def _load_custom_csv(self):
        """
        Handles file upload and extraction of features & target variable for a custom CSV dataset.
        """
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            self.X = df.iloc[:, :-1]  # All columns except the last one as features
            self.y = df.iloc[:, -1]  # Last column as target

