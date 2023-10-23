import pandas as pd


def load_data(train_path, test_path, dtype={}):
    """
    Load train and test data from csv files
    :param train_path:
    :param test_path:
    :param dtype:
    :return: train_df, test_df
    """
    train_df = pd.read_csv(train_path, dtype=dtype)
    test_df = pd.read_csv(test_path, dtype=dtype)
    return train_df, test_df


def display_dataset_info(df):
    """
    Display basic dataset information.
    """
    print("Data set Information")
    print(df.info())
    print("\nDuplicate rows:", df.duplicated().sum())


def drop_duplicates(df):
    """
    Remove duplicate rows from a dataframe
    :param df:
    :return:
    """
    return df.drop_duplicates(inplace=True)


def handle_missing_values(df):
    """
    Handle missing values in a dataframe
    :param df:
    :return:
    """
    # Replace with your own missing value handling logic if necessary
    return df.dropna(inplace=True)
