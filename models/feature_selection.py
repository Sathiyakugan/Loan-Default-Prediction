import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_important_features(X, y):
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X, y)
    feature_importances = rf_selector.feature_importances_
    features_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
    features_df = features_df.sort_values(by="Importance", ascending=False)
    return features_df


def split_data(X, y):
    """
    Split data into train and test sets.

    X: features
    y: target
    :return X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
