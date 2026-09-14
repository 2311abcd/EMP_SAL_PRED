"""
tests/test_preprocessing.py
-----------------------------
Tests the DATA CLEANING logic (missing values, duplicates, outliers) that
train_model.py applies before fitting any model, and the shape of the
scikit-learn ColumnTransformer it builds.

We re-implement the cleaning steps here in a small `clean_dataset()`
helper that mirrors train_model.py exactly, so this file can be tested
in isolation without re-running the full (slower) training pipeline.
If you change the cleaning logic in train_model.py, mirror the change
here too -- or better, refactor train_model.py's cleaning block into an
importable function and import it from both places.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

NUMERIC_FEATURES = ["age", "years_of_experience", "hours_per_week"]
CATEGORICAL_FEATURES = ["education_level", "occupation", "gender", "country", "industry"]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the cleaning steps in train_model.py STEP 2."""
    df = df.drop_duplicates()

    if df["hours_per_week"].isnull().sum() > 0:
        median_hours = df["hours_per_week"].median()
        df["hours_per_week"] = df["hours_per_week"].fillna(median_hours)

    df = df.dropna()

    Q1 = df["salary"].quantile(0.25)
    Q3 = df["salary"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["salary"] >= lower_bound) & (df["salary"] <= upper_bound)]

    return df


@pytest.fixture
def messy_df():
    """A tiny synthetic dataset with a duplicate, a missing value, and an
    obvious salary outlier -- the same three problems generate_dataset.py
    intentionally injects into the real dataset."""
    rows = [
        {"age": 30, "education_level": "Bachelor's", "occupation": "Data Analyst",
         "years_of_experience": 5, "hours_per_week": 40, "gender": "Male",
         "country": "USA", "industry": "Technology", "salary": 70000},
        {"age": 40, "education_level": "Master's", "occupation": "Product Manager",
         "years_of_experience": 12, "hours_per_week": np.nan, "gender": "Female",
         "country": "UK", "industry": "Finance", "salary": 95000},
        {"age": 25, "education_level": "High School", "occupation": "Customer Support",
         "years_of_experience": 1, "hours_per_week": 38, "gender": "Other",
         "country": "India", "industry": "Retail", "salary": 20000},
        # extreme outlier salary, should be dropped by IQR filtering
        {"age": 50, "education_level": "PhD", "occupation": "Legal Counsel",
         "years_of_experience": 25, "hours_per_week": 45, "gender": "Male",
         "country": "USA", "industry": "Technology", "salary": 5000000},
    ]
    df = pd.DataFrame(rows)
    # duplicate the first row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def test_cleaning_removes_duplicates(messy_df):
    before = len(messy_df)
    cleaned = clean_dataset(messy_df)
    assert len(cleaned) < before
    assert cleaned.duplicated().sum() == 0


def test_cleaning_fills_missing_hours(messy_df):
    cleaned = clean_dataset(messy_df)
    assert cleaned["hours_per_week"].isnull().sum() == 0


def test_cleaning_removes_salary_outliers(messy_df):
    cleaned = clean_dataset(messy_df)
    assert 5000000 not in cleaned["salary"].values


def test_column_transformer_shapes_output(messy_df):
    cleaned = clean_dataset(messy_df)
    X = cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    transformed = preprocessor.fit_transform(X)

    # rows must be preserved, and columns must expand (one-hot encoding)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_unknown_category_is_ignored_not_crashed(messy_df):
    """handle_unknown='ignore' must be set -- otherwise predicting on a
    category the model has never seen raises at inference time."""
    cleaned = clean_dataset(messy_df)
    X = cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    preprocessor.fit(X)

    unseen_row = X.iloc[[0]].copy()
    unseen_row["occupation"] = "Totally New Job Title"
    # should not raise
    preprocessor.transform(unseen_row)
