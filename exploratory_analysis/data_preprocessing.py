import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer that selects specific columns.

    Attributes:
    - columns (list): List of column names to select.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """No fitting necessary, returns self."""
        return self

    def transform(self, X):
        """Selects and returns the specified columns from the input dataframe."""
        return X[self.columns]


class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.edu_employ_mapping = {}

    def fit(self, X, y=None):
        education_vals = X["Education"].astype(str).unique()
        employment_vals = X["EmploymentType"].astype(str).unique()

        # Generate all possible combinations and map to a unique integer
        all_combinations = [f"{edu}_{employ}" for edu, employ in itertools.product(education_vals, employment_vals)]
        self.edu_employ_mapping = {combo: idx for idx, combo in enumerate(all_combinations)}
        return self

    def transform(self, X, y=None):
        # Binning Age
        bins = [18, 30, 45, 60, 100]
        labels = ["Young", "Middle_Aged", "Senior", "Elder"]
        X["AgeGroup"] = pd.cut(X["Age"], bins=bins, labels=labels, right=False)

        # Polynomial Feature: Squared Income
        X["Income_Squared"] = X["Income"] ** 2

        # Interaction term: Income/CreditScore
        X["Income_per_CreditScore"] = X["Income"] / X["CreditScore"]

        # Frequency Encoding for Education
        education_freq = X["Education"].value_counts(normalize=True).to_dict()
        X["Education_Freq"] = X["Education"].map(education_freq)
        X["Education_Freq"] = X["Education_Freq"].astype('float16')

        # Feature Crossing: Education with EmploymentType (One-hot encoding handled in DataEncoder)
        X["Edu_Employ"] = X["Education"].astype(str) + "_" + X["EmploymentType"].astype(str)

        for col in ['AgeGroup', 'Edu_Employ']:
            X[col] = X[col].astype('category')

        return X


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies logarithmic transformation to specific columns.

    Attributes:
    - columns (list): List of column names to apply the transformation on.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        """No fitting necessary, returns self."""
        return self

    def transform(self, X):
        """Applies logarithmic transformation to specified columns."""

        if not self.columns:
            self.columns = [col for col in X.columns if
                            X[col].dtype in [np.float64, np.int64, np.int16, np.int8, np.int32, np.int64]]

        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col] = np.log1p(X_transformed[col])
        return X_transformed


class DataEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer that encodes categorical columns using label encoding.

    Attributes:
    - columns (list): List of categorical column names to encode.
    - label_encoders (dict): Dictionary to store label encoders for each column.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        # if no columns provided, find columns with datatype "category"
        if not self.columns:
            self.columns = [col for col in X.columns if X[col].dtype == "category"]

        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders.items():
            print(col)
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders.items():
            X_copy[col] = encoder.inverse_transform(X_copy[col])
        return X_copy


class DataScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer that scales specified columns using standard scaling.

    Attributes:
    - columns (list): List of column names to scale.
    - scaler (StandardScaler): Scaler object to apply standard scaling.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """Fits the scaler using the specified columns."""
        if not self.columns:
            self.columns = [col for col in X.columns if
                            X[col].dtype in [np.float64, np.int64, np.int16, np.int8, np.int32, np.int64]]
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        """Applies standard scaling using the fitted scaler."""
        X_scaled = X.copy()
        X_scaled[self.columns] = self.scaler.transform(X_scaled[self.columns])
        return X_scaled


class HandleMissingValues(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Assuming numerical columns are filled with median and categorical with mode
        # This can be customized based on specific dataset and strategy
        for col in X.columns:
            if X[col].dtype == "category":
                X[col].fillna(X[col].mode()[0], inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)
        return X


class RemoveDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.drop_duplicates(inplace=True)
        return X
