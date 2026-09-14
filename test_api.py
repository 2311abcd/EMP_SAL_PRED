"""
tests/test_api.py
-------------------
End-to-end tests for the FastAPI endpoints in main.py, using FastAPI's
TestClient (which needs `httpx` installed -- it's in requirements.txt).

Includes a dedicated regression test for the PDF report bug: generating
a report for a profile whose industry contains an "&" (e.g.
"Media & Entertainment") used to crash reportlab's Paragraph parser.

Run with:
    pytest tests/test_api.py -v
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # main.py loads best_model.pkl / index.html via relative paths

import main  # noqa: E402

client = TestClient(main.app)

VALID_PAYLOAD = {
    "age": 32,
    "education_level": "Bachelor's",
    "occupation": "Software Engineer",
    "years_of_experience": 8,
    "hours_per_week": 42,
    "gender": "Female",
    "country": "USA",
    "industry": "Technology",
}


def test_homepage_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_success():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_salary"] > 0
    assert "salary_range" in body
    assert "experience_level" in body


def test_predict_rejects_invalid_occupation():
    payload = {**VALID_PAYLOAD, "occupation": "Astronaut"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_age():
    payload = {**VALID_PAYLOAD, "age": 5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_generate_report_returns_pdf():
    response = client.post("/generate-report", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    # a real PDF always starts with this magic header
    assert response.content[:5] == b"%PDF-"


def test_generate_report_with_ampersand_in_industry_does_not_crash():
    """Regression test: this exact payload used to make /generate-report
    return a 500 error because reportlab's Paragraph() parses its input
    as mini-XML, and an unescaped '&' broke the parser."""
    payload = {**VALID_PAYLOAD, "industry": "Media & Entertainment"}
    response = client.post("/generate-report", json=payload)
    assert response.status_code == 200
    assert response.content[:5] == b"%PDF-"


def test_generate_report_with_special_characters_everywhere():
    payload = {
        **VALID_PAYLOAD,
        "occupation": "Business Analyst",
        "country": "UAE",
        "industry": "Media & Entertainment",
    }
    response = client.post("/generate-report", json=payload)
    assert response.status_code == 200
    assert response.content[:5] == b"%PDF-"


def test_what_if_returns_current_and_new_salary():
    response = client.post("/what-if", json={
        "employee": VALID_PAYLOAD,
        "experience_change": 3,
        "hours_change": 0,
    })
    assert response.status_code == 200
    body = response.json()
    assert "current_salary" in body
    assert "new_salary" in body
    assert "difference" in body


def test_feature_importance_returns_all_factors():
    response = client.post("/feature-importance", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["supported"] is True
    assert len(body["top_factors"]) == len(main.FEATURE_COLUMNS)


def test_model_info_returns_metrics():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "best_model_name" in body
    assert "model_metrics" in body
