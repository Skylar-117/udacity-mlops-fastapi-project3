"""Test API

Author: Dan Sun
Date: 2022-01-07
"""
import pytest
import src.api as api

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Get user data
    """
    api_client = TestClient(api.app)
    return api_client


def test_get_home_message(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Bonjour!"}


def test_get_malformed(client):
    r = client.get("/others")
    assert r.status_code != 200


def test_post_positive(client):
    r = client.post("/", json={
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
        "hours_per_week": 46})
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_negative(client):
    r = client.post("/", json={
        "workclass": "Private",
        "education": "HS-grad",
        "marital_status": "Divorced",
        "occupation": "Craft-repair",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
        "age": 34,
        "education_num": 9,
        "hours_per_week": 40})
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_malformed(client):
    r = client.post("/", json={
        "workclass": "Local-gov",
        "education": "HS-grad",
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Unmarried",
        "race": "ERROR",
        "sex": "Female",
        "native_country": "United-States",
        "age": 34,
        "education_num": 9,
        "hours_per_week": 33})
    assert r.status_code == 422
