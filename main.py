from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from exploratory_analysis.data_cleaning import load_data
from exploratory_analysis.data_preprocessing import RemoveDuplicates, HandleMissingValues, DataTransformer, DataEncoder, \
    DataScaler, FeatureSelector
from models.feature_selection import split_data
from models.predict_model import predict, save_predictions
from models.train_model import train_best_model

# Data Paths
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

dtype = {
    'Age': 'int8',
    'Income': 'int32',
    'LoanAmount': 'int32',
    'CreditScore': 'int16',
    'MonthsEmployed': 'int8',
    'NumCreditLines': 'int8',
    'InterestRate': 'float16',
    'LoanTerm': 'int8',
    'DTIRatio': 'float16',
    'Default': 'int8',
    'Education': 'category',
    'EmploymentType': 'category',
    'MaritalStatus': 'category',
    'HasMortgage': 'category',
    'HasDependents': 'category',
    'LoanPurpose': 'category',
    'HasCoSigner': 'category'
}

# Selected features based on the EDA
features_to_consider = [
    "CreditScore",
    "NumCreditLines",
    "Income",
    "MonthsEmployed",
    "LoanAmount",
    "DTIRatio",
    "InterestRate",
    "Age",
    "LoanTerm"
]


def main():
    # Load Data
    print("Loading data...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH, dtype=dtype)

    print(train_df.info())

    # Define preprocessing pipeline
    print("Defining preprocessing pipeline...")
    preprocessing_pipeline = ImbPipeline([
        ('feature_selector', FeatureSelector(columns=features_to_consider)),
        ('remove_duplicates', RemoveDuplicates()),
        ('handle_missing', HandleMissingValues()),
        ('transformer', DataTransformer()),
        ('scaler', DataScaler()),
        ('encoder', DataEncoder()),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),  # This step will oversample the minority class
    ])

    # Split data into features and target first (as SMOTE works on this)
    X = train_df.drop(columns=['LoanID', 'Default'])
    y = train_df['Default']

    # Preprocess data
    X_preprocessed, y_resampled = preprocessing_pipeline.fit_resample(X, y)
    preprocessed_test = preprocessing_pipeline[:-1].transform(
        test_df.drop(columns=['LoanID']))  # Avoid applying SMOTE on test data

    X_preprocessed.info()

    # # Split data into training and testing sets
    print("Splitting data...")
    X_train, X_val, y_train, y_val = split_data(X_preprocessed, y_resampled)

    print("Training model...")
    model = train_best_model(X_train, y_train)

    # Prediction
    print("Predicting...")
    predicted_probability = predict(model, preprocessed_test)
    print("Saving predictions...")
    save_predictions(test_df["LoanID"], predicted_probability, "data/prediction_submission.csv")

    print("Successfully executed the pipeline and predictions saved!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
