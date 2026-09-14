"""
tests/test_dataset.py
----------------------
Tests for dataset.csv (produced by generate_dataset.py).

Run with:
    pytest tests/test_dataset.py -v

These tests do NOT call the trained model -- they only check that the
dataset itself is well-formed: the right columns, sane value ranges, and
that the "messy data on purpose" (missing values + duplicates) that
train_model.py is supposed to clean is actually present before cleaning.
"""

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import generate_dataset as gen  # noqa: E402  (imported after sys.path tweak)

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset.csv")

EXPECTED_COLUMNS = [
    "age", "education_level", "occupation", "years_of_experience",
    "hours_per_week", "gender", "country", "industry", "salary",
]


@pytest.fixture(scope="module")
def raw_df():
    assert os.path.exists(DATASET_PATH), (
        "dataset.csv not found. Run `python generate_dataset.py` first."
    )
    return pd.read_csv(DATASET_PATH)


def test_columns_match_expected_schema(raw_df):
    assert list(raw_df.columns) == EXPECTED_COLUMNS


def test_has_rows(raw_df):
    assert len(raw_df) > 1000


def test_age_within_bounds(raw_df):
    assert raw_df["age"].min() >= 18
    assert raw_df["age"].max() <= 65


def test_salary_is_positive_and_reasonable(raw_df):
    assert (raw_df["salary"] > 0).all()
    assert raw_df["salary"].max() <= 300000


def test_categorical_values_match_generator_lists(raw_df):
    assert set(raw_df["education_level"].unique()) <= set(gen.education_levels)
    assert set(raw_df["occupation"].unique()) <= set(gen.occupations)
    assert set(raw_df["gender"].unique()) <= set(gen.genders)
    assert set(raw_df["country"].unique()) <= set(gen.countries)
    assert set(raw_df["industry"].unique()) <= set(gen.industries)


def test_expanded_category_coverage():
    """Guards the 'add more occupations/countries/industries' feature --
    fails if someone accidentally shrinks the lists back down."""
    assert len(gen.occupations) >= 20
    assert len(gen.countries) >= 12
    assert len(gen.industries) >= 12


def test_dataset_has_intentional_missing_values(raw_df):
    # generate_dataset.py deliberately injects missing hours_per_week values
    # so train_model.py has real preprocessing work to do.
    assert raw_df["hours_per_week"].isnull().sum() > 0


def test_dataset_has_intentional_duplicates(raw_df):
    assert raw_df.duplicated().sum() > 0
