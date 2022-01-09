"""FastAPI

Author: Dan Sun
Date: 2022-01-07
"""
import os
import joblib
import numpy as np
import pandas as pd
import src.utils as u

# Literal types let you indicate that an expression is equal to some specific
# primitive value. For example, if we annotate a variable with type
# Literal["foo"], this .py script will understand that variable is not only of
# type str, but is also equal to specifically the string "foo".
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel


# Declare the data object with its components and their type.
class User(BaseModel):
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    sex: Literal['Male', 'Female']
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico',
        'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines',
        'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos',
        'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador',
        'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan',
        'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland',
        'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland',
        'Hungary', 'Holand-Netherlands']
    age: int
    education_num: int
    hours_per_week: int


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Bonjour!"}


@app.post("/")
async def inference(user: User):

    # Load estimator and encoders:
    model = joblib.load("./model/model.joblib")
    cat_encoder = joblib.load("./model/ohe.joblib")
    label_binarizer = joblib.load("./model/lb.joblib")

    # Get user data into numpy array:
    user_arr = np.array([[
        user.workclass,
        user.education,
        user.marital_status,
        user.occupation,
        user.relationship,
        user.race,
        user.sex,
        user.native_country,
        user.age,
        user.education_num,
        user.hours_per_week]])

    # Get user data into pandas dataframe:
    df_temp = pd.DataFrame(data=user_arr, columns=[
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

    # Process user data:
    X, _, _, _ = u.process_data(
                df=df_temp,
                cat_features=u.get_categorical_features(),
                num_features=u.get_numerical_features(),
                training=False,
                cat_encoder=cat_encoder,
                label_binarizer=label_binarizer)

    # Run inference to generate prediction:
    y_pred = u.inference(model, X)
    y_pred_label = label_binarizer.inverse_transform(y_pred)[0]

    return {"prediction": y_pred_label}
