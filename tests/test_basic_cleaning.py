"""Test basic cleaning module

Author: Dan Sun
Date: 2022-01-07
"""
import pytest
import pandas as pd

from src.basic_cleaning import clean_data


@pytest.fixture
def data():
    """Obtain dataset
    """
    df = pd.read_csv("./data/raw_data/raw_census.csv", skipinitialspace=True)
    df = clean_data(df)
    return df


def test_null(data):
    """Check that clean data has no null values
    """
    assert data.shape == data.dropna().shape


def test_question_mark(data):
    """Check that clean data has no question mark values
    """
    assert "?" not in data.values


def test_removed_columns(data):
    """Check that clean data has successfully dropped useless columns
    """
    assert "fnlgt" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns
