"""Inference pipeline

Author: Dan Sun
Date: 2022-01-07
"""
import logging
import joblib
import pandas as pd
import src.utils as u

from sklearn.model_selection import train_test_split


def inference_score(df,
                    model_pth,
                    cat_encoder_pth,
                    label_binarizer_pth,
                    slice_metrics_pth):
    """Calculate inference score on sliced data

    For simplicity, the function just output the performance on slices of
    categorical features.

    Parameters
    ----------
    df: pandas dataframe
        Cleaned dataset.
    cat_encoder_pth: string
        Path of the pre-trained categorical encoder.
    label_binarizer_pth: string
        Path of the pre-trained label binarizer.
    slice_metrics_pth: string
        Path to save the metrics on sliced data.
    """
    # Split dataset into training and validation set:
    _, df_valid = train_test_split(df, test_size=0.20)

    # Load pre-trained eatimators:
    model = joblib.load(model_pth)
    ohe = joblib.load(cat_encoder_pth)
    lb = joblib.load(label_binarizer_pth)

    # Calculate model performance on sliced categorical features:
    slice_values = []
    cat_feats = u.get_categorical_features()
    for cat_feat in cat_feats:
        for category in df_valid[cat_feat].unique():

            # Grab rows within the same category:
            df_temp = df_valid[df_valid[cat_feat] == category]

            X_valid, y_valid, _, _ = u.process_data(
                df=df_temp,
                cat_features=u.get_categorical_features(),
                num_features=u.get_numerical_features(),
                training=False,
                cat_encoder=ohe,
                label_binarizer=lb
            )

            # Generate predictions on validation set:
            y_pred = model.predict(X_valid)

            # Calculate metrics on validation set:
            acc, recall, precision = u.calculate_metrics(
                y_valid, y_pred)

            # Log model performance on sliced data:
            _ = (f"[{cat_feat} - {category}], Accuracy={acc:.3f}, "
                 f"Recall={recall:.3f}, Precision={precision:.3f}")
            logging.info(_)
            slice_values.append(_)

    # Log sliced metrics value into a txt file:
    with open(slice_metrics_pth, "w") as f:
        for v in slice_values:
            f.write(v + "\n")


def execute():
    """Execute inference pipeline
    """
    # Set up paths:
    CLEAN_DATA_PATH = "data/clean_data/clean_census.csv"
    SCORE_TXT_PATH = "model/slice_metrics.txt"
    MODEL_PATH = "model/model.joblib"
    CAT_ENCODER_PATH = "model/ohe.joblib"
    LABEL_BINARIZER_PATH = "model/lb.joblib"

    # Load clean data:
    CLEAN_DATA = pd.read_csv(CLEAN_DATA_PATH, skipinitialspace=True)

    # Execute inference pipeline:
    inference_score(
        df=CLEAN_DATA,
        model_pth=MODEL_PATH,
        cat_encoder_pth=CAT_ENCODER_PATH,
        label_binarizer_pth=LABEL_BINARIZER_PATH,
        slice_metrics_pth=SCORE_TXT_PATH
    )


if __name__ == "__main__":
    execute()
