"""Test utils module

Author: Dan Sun
Date: 2022-01-07
"""
import pytest
import joblib
import numpy as np
import pandas as pd

import src.utils as u


@pytest.fixture
def data():
    """Obtain dataset
    """
    df = pd.read_csv("data/clean_data/clean_census.csv",
                     skipinitialspace=True)
    return df


def test_process_data(data):
    """Check that split have same number of rows for X and y
    """
    cat_encoder = joblib.load("model/ohe.joblib")
    label_binarizer = joblib.load("model/lb.joblib")

    X_valid, y_valid, _, _ = u.process_data(
        df=data,
        cat_features=u.get_categorical_features(),
        num_features=u.get_numerical_features(),
        training=False,
        cat_encoder=cat_encoder,
        label_binarizer=label_binarizer
    )

    assert len(X_valid) == len(y_valid)


def test_inference_pos():
    """Check inference on positive class
    """
    model = joblib.load("model/model.joblib")
    cat_encoder = joblib.load("model/ohe.joblib")
    label_binarizer = joblib.load("model/lb.joblib")

    # Test sample:
    arr = np.array([["Private",
                     "Masters",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Wife",
                     "White",
                     "Male",
                     "United-States",
                     53,
                     14,
                     40]])

    # Test dataframe:
    df_temp = pd.DataFrame(data=arr, columns=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "age",
        "education-num",
        "hours-per-week"])

    # Process data:
    X, _, _, _ = u.process_data(
                df=df_temp,
                cat_features=u.get_categorical_features(),
                num_features=u.get_numerical_features(),
                training=False,
                cat_encoder=cat_encoder,
                label_binarizer=label_binarizer)

    # Generate prediction on the test dataframe:
    y_pred = u.inference(model, X)
    y_pred_label = label_binarizer.inverse_transform(y_pred)[0]

    assert y_pred_label == ">50K"


def test_inference_neg():
    """Check inference on negative class
    """
    model = joblib.load("model/model.joblib")
    cat_encoder = joblib.load("model/ohe.joblib")
    label_binarizer = joblib.load("model/lb.joblib")

    # Test sample:
    arr = np.array([["Private",
                     "HS-grad",
                     "Divorced",
                     "Craft-repair",
                     "Not-in-family",
                     "White",
                     "Male",
                     "United-States",
                     34,
                     9,
                     40]])

    # Test dataframe:
    df_temp = pd.DataFrame(data=arr, columns=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "age",
        "education-num",
        "hours-per-week"])

    # Process data:
    X, _, _, _ = u.process_data(
                df=df_temp,
                cat_features=u.get_categorical_features(),
                num_features=u.get_numerical_features(),
                training=False,
                cat_encoder=cat_encoder,
                label_binarizer=label_binarizer)

    # Generate prediction on the test dataframe:
    y_pred = u.inference(model, X)
    y_pred_label = label_binarizer.inverse_transform(y_pred)[0]

    assert y_pred_label == "<=50K"
