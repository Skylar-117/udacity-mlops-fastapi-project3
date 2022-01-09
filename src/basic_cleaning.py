"""Data cleaning pipeline

Author: Dan Sun
Data: 2022-01-07
"""
import numpy as np
import pandas as pd


def clean_data(df):
    """Clean raw data

    Parameters
    ----------
    df: pandas dataframe
        This is the raw dataset.
    """
    # Drop rows containing "?" values:
    df.replace(to_replace="?", value=np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)

    # Drop columns containing too many zero values:
    df.drop(columns=["capital-loss", "capital-gain"], inplace=True)

    # Drop column "fnlgt" which seems like an ID column:
    df.drop(columns=["fnlgt"], inplace=True)

    return df


def execute():
    """Execute basic data cleaning pipeline
    """
    # Set up paths:
    RAW_DATA_PATH = "./data/raw_data/raw_census.csv"
    CLEAN_DATA_PATH = "./data/clean_data/clean_census.csv"

    # Load raw data:
    RAW_DATA = pd.read_csv(RAW_DATA_PATH, skipinitialspace=True)

    # Clean raw data:
    CLEAN_DATA = clean_data(df=RAW_DATA)

    # Save clean data:
    CLEAN_DATA.to_csv(CLEAN_DATA_PATH, index=False)


if __name__ == "__main__":
    execute()
