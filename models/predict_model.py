import pandas as pd

def predict(model, X_test):
    predicted_probability = model.predict_proba(X_test)[:, 1]
    return predicted_probability

def save_predictions(loan_ids, predicted_probabilities, file_name):
    prediction_df = pd.DataFrame({'LoanID': loan_ids, 'predicted_probability': predicted_probabilities})
    prediction_df.to_csv(file_name, index=False)
