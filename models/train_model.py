from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier





def train_best_model(X_train, y_train):
    """
    Train the best model found by GridSearchCV.
    :param X_train:
    :param y_train:
    :return: best model
    """


    # Define a pipeline. Placeholders for the 'classifier' and its 'params' will be replaced.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier())  # Placeholder
    ])

    # Logistic Regression params
    log_params = {
        "classifier": [LogisticRegression(max_iter=1000)],
        "classifier__C": [0.001, 0.01, 0.1, 1, 10],
    }

    # Random Forest params
    rf_params = {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
    }

    # XGBoost params
    xgb_params = {
        "classifier": [XGBClassifier()],
        "classifier__learning_rate": [0.01, 0.1],
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__reg_lambda": [1, 10, 100],
    }

    # CatBoost params
    cat_params = {
        "classifier": [CatBoostClassifier(verbose=False)],
        # Add any specific parameters for CatBoost you'd like to tune
    }

    # KNeighbors params
    knn_params = {
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": [3, 5, 7, 9]
    }

    # GaussianNB doesn't have parameters to tune in this context
    gnb_params = {
        "classifier": [GaussianNB()],
    }

    # Combine all parameters
    search_space = [
        log_params,
        rf_params,
        xgb_params,
        cat_params,
        knn_params,
        gnb_params
    ]

    # Grid Search with cross-validation
    grid_search = GridSearchCV(
        pipe,
        search_space,
        cv=5,
        scoring=make_scorer(roc_auc_score, needs_proba=True),  # AUC as scoring
        n_jobs=-1,  # Use all CPU cores
        verbose=10  # Display progress. Increase for more verbosity.
    )

    best_model = grid_search.fit(X_train, y_train)

    print("Best AUC:", best_model.best_score_)
    print("Best parameters:", best_model.best_params_)

    return best_model.best_estimator_