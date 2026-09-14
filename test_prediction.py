"""
tests/test_prediction.py
--------------------------
Tests the TRAINED PIPELINE (best_model.pkl) and the pure prediction-support
helper functions in main.py: get_salary_range, get_experience_level,
get_salary_insight, and compute_personalized_factors.

Requires best_model.pkl and model_metrics.json to already exist (run
`python generate_dataset.py` then `python train_model.py` first).

Run with:
    pytest tests/test_prediction.py -v
"""

import json
import os
import sys

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # main.py loads best_model.pkl / model_metrics.json via relative paths

import main  # noqa: E402


@pytest.fixture(scope="module")
def sample_employee():
    return main.EmployeeData(
        age=30,
        education_level="Master's",
        occupation="Data Scientist",
        years_of_experience=6,
        hours_per_week=42,
        gender="Female",
        country="Germany",
        industry="Technology",
    )


def test_model_pipeline_loads():
    assert main.model_pipeline is not None


def test_predict_salary_returns_positive_float(sample_employee):
    salary = main.predict_salary(sample_employee)
    assert isinstance(salary, float)
    assert salary > 0


def test_predict_salary_is_deterministic(sample_employee):
    first = main.predict_salary(sample_employee)
    second = main.predict_salary(sample_employee)
    assert first == second


def test_more_experience_generally_increases_salary(sample_employee):
    junior = sample_employee.model_copy(deep=True)
    junior.years_of_experience = 0

    senior = sample_employee.model_copy(deep=True)
    senior.years_of_experience = 20

    assert main.predict_salary(senior) >= main.predict_salary(junior)


def test_employee_to_dataframe_has_correct_columns(sample_employee):
    df = main.employee_to_dataframe(sample_employee)
    assert list(df.columns) == main.FEATURE_COLUMNS
    assert len(df) == 1


@pytest.mark.parametrize("years,expected", [
    (0, "Entry-Level"),
    (1.9, "Entry-Level"),
    (2, "Mid-Level Professional"),
    (5.9, "Mid-Level Professional"),
    (6, "Senior Professional"),
    (11.9, "Senior Professional"),
    (12, "Expert / Leadership Level"),
    (30, "Expert / Leadership Level"),
])
def test_get_experience_level_buckets(years, expected):
    assert main.get_experience_level(years) == expected


def test_get_salary_range_brackets_the_prediction():
    salary_range = main.get_salary_range(100000)
    assert salary_range["minimum"] <= 100000 <= salary_range["maximum"]
    assert salary_range["minimum"] >= 0


def test_get_salary_insight_position_labels():
    average = main.model_metrics["average_salary"]
    above = main.get_salary_insight(average * 1.5)
    below = main.get_salary_insight(average * 0.5)
    around = main.get_salary_insight(average)

    assert above["position"] == "Above Average"
    assert below["position"] == "Below Average"
    assert around["position"] == "Around Average"


def test_compute_personalized_factors_returns_all_features(sample_employee):
    predicted = main.predict_salary(sample_employee)
    factors = main.compute_personalized_factors(sample_employee, predicted)

    assert len(factors) == len(main.FEATURE_COLUMNS)
    feature_names = {f["feature"] for f in factors}
    assert feature_names == set(main.FEATURE_COLUMNS)

    # sorted by absolute impact, descending
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)


def test_valid_lists_match_model_metrics_categories():
    """Guards against the API accepting/rejecting categories the trained
    model doesn't actually know about (or vice versa)."""
    with open(os.path.join(PROJECT_ROOT, "model_metrics.json")) as f:
        metrics = json.load(f)
    # every category the API says is valid must be one the dataset generator
    # (and therefore the trained model) actually produced
    assert metrics["categorical_features"] == main.CATEGORICAL_FEATURES


def test_special_characters_in_category_do_not_break_dataframe_build():
    """Regression test for the PDF-report bug: categories containing
    special characters (e.g. 'Media & Entertainment') must flow through
    normally as plain data -- the escaping happens later, at PDF-render
    time, not here."""
    employee = main.EmployeeData(
        age=28, education_level="Bachelor's", occupation="Content Writer",
        years_of_experience=3, hours_per_week=40, gender="Other",
        country="Singapore", industry="Media & Entertainment",
    )
    df = main.employee_to_dataframe(employee)
    assert df.at[0, "industry"] == "Media & Entertainment"
    salary = main.predict_salary(employee)
    assert salary > 0


def test_numeric_columns_are_float64():
    """Regression test for a real crash on modern pandas (2.x/3.x):
    if 'age' stays int64 in the built DataFrame, writing the float
    'typical' reference value into it inside compute_personalized_factors
    raises `TypeError: Invalid value ... for dtype 'int64'`
    (pandas LossySetitemError). All numeric feature columns must be
    float64 so that swap-in never fails."""
    employee = main.EmployeeData(
        age=45, education_level="PhD", occupation="Research Scientist",
        years_of_experience=15, hours_per_week=45, gender="Male",
        country="Japan", industry="Pharmaceuticals",
    )
    df = main.employee_to_dataframe(employee)
    for numeric_col in main.NUMERIC_FEATURES:
        assert str(df[numeric_col].dtype) == "float64"


def test_compute_personalized_factors_does_not_crash_on_int_age():
    """End-to-end regression test for the exact bug: age is an int in the
    Pydantic model, but REFERENCE_VALUES['age'] (the training-set average)
    is a float -- computing personalized factors used to crash here."""
    employee = main.EmployeeData(
        age=22, education_level="High School", occupation="Customer Support",
        years_of_experience=0.5, hours_per_week=35, gender="Male",
        country="Mexico", industry="Retail",
    )
    predicted = main.predict_salary(employee)
    factors = main.compute_personalized_factors(employee, predicted)
    assert len(factors) == len(main.FEATURE_COLUMNS)
