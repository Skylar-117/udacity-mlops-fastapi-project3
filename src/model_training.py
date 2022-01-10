"""Train model pipeline

Author: Dan Sun
Date: 2022-01-07
"""
import pandas as pd
import src.utils as u
import joblib

from sklearn.model_selection import train_test_split


def train_model(df):
    """Train model

    Parameters
    ----------
    df: pandas dataframe
        Cleaned dataset.

    Returns
    -------
    model: sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    cat_encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder.
    label_binarizer: sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer.
    """
    df_train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, ohe, lb = u.process_data(
        df=df_train,
        cat_features=u.get_categorical_features(),
        num_features=u.get_numerical_features(),
    )
    cv_scores = ["accuracy", "roc_auc", "f1"]
    model = u.train_model(X_train, y_train, cv_scores)

    return model, ohe, lb


def execute():
    """Execute model training pipeline
    """
    # Set up paths:
    CLEAN_DATA_PATH = "./data/clean_data/clean_census.csv"

    # Load clean data:
    CLEAN_DATA = pd.read_csv(CLEAN_DATA_PATH, skipinitialspace=True)

    # Execute model training pipeline:
    model, ohe, lb = train_model(CLEAN_DATA)

    # Save estimator and encoders:
    joblib.dump(model, "model/model.joblib")
    joblib.dump(ohe, "model/ohe.joblib")
    joblib.dump(lb, "model/lb.joblib")


if __name__ == "__main__":
    execute()
