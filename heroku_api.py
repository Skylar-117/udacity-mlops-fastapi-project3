"""Heroku API

Author: Dan Sun
Date: 2022-01-07
"""
import requests

user_data = {
    "workclass": "State-gov",
    "education": "Doctorate",
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "native_country": "United-States",
    "age": 48,
    "education_num": 16,
    "hours_per_week": 46
}
r = requests.post("https://udacity-mlops-fastapi.herokuapp.com",
                  json=user_data)

assert r.status_code == 200

print(f"Response code: {r.status_code}")
print(f"Response body: {r.json()}")
