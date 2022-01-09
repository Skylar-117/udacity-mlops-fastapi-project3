"""Utility functions

Author: Dan Sun
Date: 2022-01-07
"""
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    roc_auc_score, f1_score


def get_categorical_features():
    """Get the name of all categorical features

    Returns
    -------
    cat_feats: list of string
        List of categorical feature names.
    """
    cat_feats = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    return cat_feats


def get_numerical_features():
    """Get the name of all numerical features

    Returns
    -------
    num_feats: list of string
        List of numerical feature names.
    """
    num_feats = [
        "age",
        "education-num",
        "hours-per-week",
    ]

    return num_feats


def _get_model_params():
    """Set model parameters

    Returns
    -------
    params: dictionary
        Dictionary of the model parameters.
    """
    params = {
        "n_estimators": 200,
        "random_state": 42,
        "max_depth": 5,
        "criterion": "entropy",
        "n_jobs": -1
    }

    return params


def process_data(df,
                 cat_features,
                 num_features,
                 training=True,
                 cat_encoder=None,
                 label_binarizer=None):
    """Process data for later train test split

    Parameters
    ----------
    df: pandas dataframe
        Original dataset.
    cat_features: list of string
        List of categorical feature names.
    num_features: list of string
        List of numerical feature names.
    training: bool, default=True
        This indicates if it is for training purpose or inference purpose.
    cat_encoder: sklearn.preprocessing._encoders.OneHotEncoder, default=None
        Trained sklearn one hot encoder. Only used if training=False.
    label_binarizer: sklearn.preprocessing._encoders.LabelBinarizer,
                     default=None
        Trained sklearn label binarizer. If the label/target is not integer,
        then use LabelBinarizer to convert string to integer. Only used if
        training=False.

    Returns
    -------
    X: numpy array
        Processed features.
    y: numpy array
        Processed label
    cat_encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training=True, otherwise returns default
        encoder.
    label_binarizer: sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns default
        binarizer.
    """
    # Asign X and y:
    feats = cat_features + num_features
    X = df[feats]
    y = df.drop(columns=feats, inplace=False)

    # Encode categorical features using one hot encoding:
    X_cat = X[cat_features]
    X_num = X[num_features]
    if (training):
        cat_encoder = OneHotEncoder()
        label_binarizer = LabelBinarizer()
        X_cat = cat_encoder.fit_transform(X_cat)
        y = label_binarizer.fit_transform(y.values).ravel()
    else:
        X_cat = cat_encoder.transform(X_cat)
        # Case where y is empty dataframe since we are doing inference
        try:
            y = label_binarizer.transform(y.values).ravel()
        except ValueError:
            pass

    # Concatenate numerical and categorical features:
    # Since we have many categorical features, X_cat will be a sparse matrix
    # which is not a subclasses of numpy arrays. Thus, numpy methods often do
    # not work. To address this, make the sparse matrix dense first using
    # `.toarray()`, then use np.concatenate().
    X_cat = X_cat.toarray()
    X = np.concatenate([X_cat, X_num], axis=1)

    return X, y, cat_encoder, label_binarizer


def train_model(X_train, y_train, cv_scores):
    """Train a machine learning model and calculate cv scores

    Parameters
    ----------
    X_train: numpy array
        Training feature data.
    y_train: numpy array
        Training label data.
    cv_scores: list of string
        Name of scores to calculate.

    Returns
    -------
    model: sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    """
    # Fit training data to model estimator:
    _params = _get_model_params()
    model = RandomForestClassifier(**_params)
    model.fit(X_train, y_train)

    # Calculate cross validated performance scores:
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    for s in cv_scores:
        cv_score = cross_val_score(model, X_train, y_train, scoring=s,
                                   cv=cv, n_jobs=-1)
        logging.info(f"{s}: {cv_score}:.2f")

    return model


def calculate_metrics(y_true, y_pred):
    """Calculate model metrics

    Parameters
    ----------
    y_true: numpy array
        Binarized true labels.
    y_pred: numpy array
        Predicted label values.
    """
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)

    return acc, recall, precision


def inference(model, X):
    """Run model inference and predict on new data

    Parameters
    ----------
    model: sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model
    X: pandas dataframe
        New dataset used to generate the prediction

    Returns
    -------
    y_pred: numpy array
        Binary predictions.
    """
    y_pred = model.predict(X)

    return y_pred
