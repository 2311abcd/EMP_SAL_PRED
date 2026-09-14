"""
train_model.py
----------------
This is the COMPLETE Machine Learning training pipeline for the
Employee Salary Prediction & Career Insights System.

What this script does, step by step:
    1. Load the dataset with Pandas
    2. Show basic dataset information (rows, columns, missing values, duplicates)
    3. Clean the data (handle missing values, remove duplicates, handle outliers)
    4. Build a Scikit-learn Pipeline (ColumnTransformer) that encodes categorical
       columns and scales numeric columns -- the SAME pipeline is reused for
       every model and later for every prediction, so training and prediction
       never go "out of sync".
    5. Train 3 regression models: Linear Regression, Random Forest, Gradient Boosting
    6. Evaluate each model with MAE, RMSE and R2 Score
    7. Automatically pick the best model (highest R2 Score)
    8. Save the winning pipeline (preprocessing + model together) as best_model.pkl
    9. Save metrics + feature information as model_metrics.json

Run this file with:
    python train_model.py
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =================================================================
# STEP 1: DATA LOADING
# =================================================================
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv("dataset.csv")

print(f"Number of rows        : {df.shape[0]}")
print(f"Number of columns     : {df.shape[1]}")
print(f"Column names          : {list(df.columns)}")
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nDuplicate records     : {df.duplicated().sum()}")


# =================================================================
# STEP 2: DATA PREPROCESSING
# =================================================================
print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

# ---- 2a. Remove duplicate records ----
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows.")

# ---- 2b. Handle missing values ----
# Numeric column 'hours_per_week' has some missing values.
# We fill them with the median, which is a safe, standard approach
# because it is not affected by extreme outliers.
if df["hours_per_week"].isnull().sum() > 0:
    median_hours = df["hours_per_week"].median()
    df["hours_per_week"] = df["hours_per_week"].fillna(median_hours)
    print(f"Filled missing 'hours_per_week' values with median = {median_hours}")

# Drop any remaining rows that still have missing values (safety net)
before = len(df)
df = df.dropna()
print(f"Dropped {before - len(df)} rows that still had missing values.")

# ---- 2c. Outlier handling ----
# We use the IQR (Interquartile Range) method on the target column 'salary'
# to remove unrealistic extreme values that could confuse the model.
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
before = len(df)
df = df[(df["salary"] >= lower_bound) & (df["salary"] <= upper_bound)]
print(f"Removed {before - len(df)} salary outliers using the IQR method.")
print(f"Final dataset size after cleaning: {len(df)} rows")

# ---- 2d. Define input features (X) and target (y) ----
TARGET_COLUMN = "salary"

NUMERIC_FEATURES = ["age", "years_of_experience", "hours_per_week"]
CATEGORICAL_FEATURES = ["education_level", "occupation", "gender", "country", "industry"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ---- 2e-bis. Compute "typical profile" reference values ----
# These represent a typical/average employee in the dataset. Later, the
# FastAPI backend uses them to explain individual predictions: for each
# feature, it swaps the user's actual value with this "typical" value and
# sees how much the prediction changes. That change IS the feature's real,
# personalized dollar impact for that specific user -- not a fixed,
# one-size-fits-all importance score.
reference_values = {}
for col in NUMERIC_FEATURES:
    reference_values[col] = round(float(X[col].mean()), 2)
for col in CATEGORICAL_FEATURES:
    reference_values[col] = X[col].mode()[0]

print("\nTypical profile (used later to explain individual predictions):")
for feature, value in reference_values.items():
    print(f"   {feature}: {value}")

# ---- 2e. Train / Test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {len(X_train)}   Testing rows: {len(X_test)}")

# ---- 2f. Build the preprocessing pipeline ----
# Categorical Encoding -> OneHotEncoder (correct choice for non-ordinal categories)
# Feature Scaling      -> StandardScaler (helps Linear Regression converge well)
#
# Using ColumnTransformer guarantees that EXACTLY the same transformation is
# applied during training and during every future prediction, which avoids
# the classic "feature mismatch" bug.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)


# =================================================================
# STEP 3: MODEL TRAINING
# =================================================================
print("\n" + "=" * 60)
print("STEP 3: TRAINING MODELS")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=18, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = {}
trained_pipelines = {}

for name, model in models.items():
    # Every model gets its own full pipeline: preprocessing + model.
    # This means the SAME preprocessing is guaranteed for every model.
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2 Score": round(r2, 4)}
    trained_pipelines[name] = pipeline

    print(f"\n{name}")
    print(f"   MAE      : {mae:,.2f}")
    print(f"   RMSE     : {rmse:,.2f}")
    print(f"   R2 Score : {r2:.4f}")


# =================================================================
# STEP 4: MODEL EVALUATION & SELECTION
# =================================================================
print("\n" + "=" * 60)
print("STEP 4: MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame(results).T
print(comparison_df)

# The best model is the one with the highest R2 Score (best fit to test data)
best_model_name = max(results, key=lambda name: results[name]["R2 Score"])
best_pipeline = trained_pipelines[best_model_name]

print(f"\nBEST MODEL SELECTED: {best_model_name}")


# =================================================================
# STEP 5: FEATURE IMPORTANCE (only for tree-based models)
# =================================================================
feature_importance_list = []

if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
    # Get the actual transformed feature names produced by the ColumnTransformer
    # (OneHotEncoder expands each category into its own column, e.g.
    #  "occupation_Software Engineer"). This step is critical -- without it,
    # importances would not line up with human-readable feature names.
    ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    importances = best_pipeline.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    # Group the one-hot-encoded columns back into their original feature
    # (e.g. all "occupation_*" columns are summed into one "occupation" score)
    def original_feature_name(encoded_name):
        for original in CATEGORICAL_FEATURES:
            if encoded_name.startswith(original + "_"):
                return original
        return encoded_name

    importance_df["original_feature"] = importance_df["feature"].apply(original_feature_name)
    grouped_importance = (
        importance_df.groupby("original_feature")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop Factors Affecting Salary:")
    for i, (feature, score) in enumerate(grouped_importance.items(), start=1):
        print(f"   {i}. {feature}  (importance: {score:.4f})")
        feature_importance_list.append({"feature": feature, "importance": round(float(score), 4)})
else:
    print(f"\n{best_model_name} does not support feature importance.")


# =================================================================
# STEP 6: SAVE THE MODEL AND METRICS
# =================================================================
print("\n" + "=" * 60)
print("STEP 6: SAVING MODEL")
print("=" * 60)

joblib.dump(best_pipeline, "best_model.pkl")
print("Saved trained pipeline to best_model.pkl")

# Compute salary standard deviation of residuals -> used later to build a
# realistic "estimated range" around each prediction (see README for details)
residuals = y_test.values - best_pipeline.predict(X_test)
residual_std = float(np.std(residuals))

metrics_output = {
    "best_model_name": best_model_name,
    "all_model_results": results,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "feature_importance": feature_importance_list,
    "residual_std": round(residual_std, 2),
    "average_salary": round(float(y.mean()), 2),
    "reference_values": reference_values,
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics_output, f, indent=4)

print("Saved model_metrics.json")
print("\nTraining complete! You can now run: uvicorn main:app --reload")
