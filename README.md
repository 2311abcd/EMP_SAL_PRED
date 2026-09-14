# Employee Salary Prediction & Career Insights System

A complete, end-to-end Machine Learning web application that predicts an
employee's estimated annual salary and provides real, model-backed career
insights — including a What-If simulator and feature importance breakdown.

---

## 1. Project Description

This project trains a real regression model on employee data (age,
education, occupation, experience, hours worked, country, and industry) and
serves live predictions through a FastAPI backend, with a clean HTML/CSS/JS
frontend. It is designed to be simple enough to explain line-by-line in an
interview, while still covering a full, realistic ML product workflow.

## 2. Problem Statement

Job seekers and employees often have no easy way to check whether their pay
is in line with their experience, education, and role — and even less way to
understand *what* would move the needle. This project solves that by:

- Predicting an estimated salary from a real profile.
- Showing a realistic salary **range**, not a falsely precise single number.
- Explaining *why* the model predicted that number (feature importance).
- Letting the user simulate changes ("What if I get 2 more years of
  experience?") and see the actual effect, computed by the same trained
  model — not guessed or hardcoded.

## 3. Features

| Feature | Description |
|---|---|
| Salary Prediction | Predicts estimated annual salary from user input |
| Salary Range | Shows a realistic min–max range around the prediction |
| Career Insights | Experience level + salary position vs. the dataset average |
| What-If Simulator | Re-runs a modified profile through the same trained pipeline |
| Improvement Suggestions | Plain-language, model-based estimate of impact |
| Feature Importance | Personalized dollar impact per factor, recalculated for every profile |
| PDF Report Download | One-click, clean PDF summary of the prediction and insights |
| Save & Compare Profiles | Save predicted profiles to the browser (local storage) and compare 2+ side by side |
| Broad Category Coverage | 25 occupations, 16 countries, 14 industries (up from 10 / 6 / 6) |

## 4. Technology Stack

**Backend / ML:** Python, Pandas, NumPy, Scikit-learn, Joblib, FastAPI, Pydantic, ReportLab (PDF generation)
**Frontend:** HTML, CSS, Vanilla JavaScript (`fetch()` — no frameworks)
**Testing:** Pytest, FastAPI's `TestClient` (via `httpx`)

No React, no Node.js, no backend database, no Docker, no authentication, and
no external AI APIs are used anywhere in this project. "Save & Compare
Profiles" uses the browser's own `localStorage` — nothing is sent to or
stored on the server.

## 5. Machine Learning Workflow

1. **Load data** with Pandas (`train_model.py`) and print shape, columns,
   missing values, and duplicate counts.
2. **Preprocess**:
   - Fill missing `hours_per_week` values with the median.
   - Drop duplicate rows.
   - Remove salary outliers using the IQR method.
   - Encode categorical columns with `OneHotEncoder`.
   - Scale numeric columns with `StandardScaler`.
   - All of this lives inside a single `ColumnTransformer` +
     `Pipeline`, so the exact same transformation is used for
     training AND every future prediction — this is what prevents
     feature-mismatch bugs.
3. **Train** three regression models: Linear Regression, Random Forest, and
   Gradient Boosting.
4. **Evaluate** each with MAE, RMSE, and R² Score on a held-out test set.
5. **Select** the best model automatically (highest R²).
6. **Save** the full pipeline (preprocessing + model together) as
   `best_model.pkl`, plus `model_metrics.json` with scores and feature
   importance.
7. **Serve** predictions live from FastAPI using the exact saved pipeline.

## 6. Dataset Information

**Important and honest note about the dataset:** most free public salary
datasets (e.g. the well-known UCI "Adult Income" dataset) only label people
as earning `<=50K` or `>50K` — that is a **classification** dataset, not a
regression one, so it cannot produce an actual dollar prediction like
"$72,500" or support a meaningful What-If simulator.

Since this project specifically needs **real salary regression**, this repo
includes `generate_dataset.py`, which builds a dataset of 6,000 employee
records using a formula grounded in real-world salary drivers — more
experience increases pay with diminishing returns, higher education level
increases pay, certain occupations/industries pay more, higher-cost countries
pay more — plus realistic random noise, missing values, and duplicate rows
so the preprocessing steps have genuine work to do.

This keeps every later step of the pipeline **100% real**: real data loading,
real cleaning, real training, real evaluation, and real predictions from an
actually-trained model. Only the data source is synthetic-but-realistic
rather than downloaded from a public host — this is a standard, transparent
approach used for portfolio projects when no free dataset matches the exact
schema needed. If you have access to a real salary dataset (e.g. from
Kaggle) with similar columns, you can drop it in as `dataset.csv` and
`train_model.py` will work with minimal changes to the column names at the
top of the script.

**Note on gender:** gender is collected as a demographic field for display
purposes only. It has **zero effect** on the generated salary formula, and
the trained model correctly learns almost no importance for it — this is a
deliberate, responsible choice to avoid the model learning or reinforcing
any pay-gap pattern.

**Target variable:** `salary` (continuous, in USD)
**Input features:** `age`, `education_level`, `occupation`,
`years_of_experience`, `hours_per_week`, `gender`, `country`, `industry`

## 7. Model Comparison

Actual results from `train_model.py` (your numbers may vary slightly since
`generate_dataset.py` uses a fixed random seed, but should be very close):

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Linear Regression (best)** | **~11,035** | **~14,221** | **~0.912** |
| Random Forest (depth-capped) | ~11,944 | ~15,814 | ~0.892 |
| Gradient Boosting | ~12,584 | ~16,585 | ~0.881 |

Your exact numbers will differ slightly after retraining (random seed keeps
it close, not identical), and the winner **can change** depending on your
scikit-learn version and the dataset size — the pipeline always picks
whichever model actually scores highest on your machine, automatically.

> **Note on model size:** `RandomForestRegressor` is deliberately capped
> (`max_depth=18`, `min_samples_leaf=3`, `n_estimators=150`) instead of left
> unbounded. With the expanded category list (25 occupations × 16 countries
> × 14 industries, one-hot encoded), an uncapped Random Forest produced a
> **~170 MB** `best_model.pkl` for a barely-better R² — not worth it for a
> project meant to run and be shared easily. If you retrain and Random
> Forest or Gradient Boosting wins on your machine, that's expected and
> fine; if the `.pkl` file balloons again, tighten `max_depth` further.

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend page |
| POST | `/predict` | Predicts salary, range, and career insights |
| POST | `/what-if` | Runs a what-if scenario through the trained model |
| GET | `/model-info` | Returns best model name, metrics, and features used |
| POST | `/feature-importance` | Returns personalized top factors for one specific profile |
| POST | `/generate-report` | Builds and streams back a downloadable PDF report |

### Example: `POST /predict`

Request body:
```json
{
  "age": 30,
  "education_level": "Bachelor's",
  "occupation": "Software Engineer",
  "years_of_experience": 5,
  "hours_per_week": 40,
  "gender": "Male",
  "country": "USA",
  "industry": "Technology"
}
```

Response:
```json
{
  "predicted_salary": 72500.0,
  "salary_range": { "minimum": 65000.0, "maximum": 80000.0 },
  "experience_level": "Mid-Level Professional",
  "salary_insight": "Above Average",
  "salary_insight_message": "Your profile predicts a salary above the typical average in our dataset.",
  "career_insight": "You are currently classified as a 'Mid-Level Professional' based on 5.0 years of experience. Your experience, education and industry combine to place you above the typical average."
}
```

### Example: `POST /what-if`

Request body:
```json
{
  "employee": { "age": 30, "education_level": "Bachelor's", "occupation": "Software Engineer",
                "years_of_experience": 5, "hours_per_week": 40, "gender": "Male",
                "country": "USA", "industry": "Technology" },
  "experience_change": 2,
  "hours_change": 0
}
```

Response:
```json
{
  "current_salary": 72500.0,
  "new_salary": 79800.0,
  "difference": 7300.0,
  "scenario": "Scenario: increasing experience by 2.0 year(s)",
  "explanation": "Model-based estimate: increasing experience by 2.0 year(s) could change the estimated salary by approximately $7,300.00 (increase). This is an estimate from the trained model, not a guarantee."
}
```

### `POST /feature-importance`

Unlike a typical "feature importance" endpoint, this is **personalized per
request** — it does not return the same fixed list for every user. It takes
the same body shape as `/predict`, and for every field it swaps in the
"typical" value seen in training (the dataset average for numeric fields,
the most common category for categorical fields), re-runs the trained
pipeline, and reports the real dollar difference that field makes for
*this specific profile*. Change any input and the list of factors, their
order, and their dollar amounts change with it.

Request body: same shape as `/predict`.

Response:
```json
{
  "supported": true,
  "best_model_name": "Linear Regression",
  "predicted_salary": 72500.0,
  "top_factors": [
    { "feature": "country", "impact": 18320.5, "direction": "positive", "your_value": "USA", "typical_value": "Canada" },
    { "feature": "education_level", "impact": 9410.2, "direction": "positive", "your_value": "Bachelor's", "typical_value": "Bachelor's" },
    { "feature": "occupation", "impact": -2100.0, "direction": "negative", "your_value": "Software Engineer", "typical_value": "Financial Analyst" }
  ]
}
```

### `POST /generate-report`

Accepts the same body shape as `/predict`, but instead of returning JSON it
recomputes the prediction internally and streams back a PDF file
(`Content-Type: application/pdf`, with a `Content-Disposition: attachment`
header so the browser downloads it directly). The report includes the
employee's profile, the predicted salary and range, career insights, and
the personalized top factors (same method as `/feature-importance`) —
nothing is saved on the server; the PDF is built in memory per request.

## 9. How the Salary Range Is Calculated

The range is **not** a formal statistical confidence interval — it is a
clearly documented approximation. During training, we measure the *residual
standard deviation* (how far off the model's test-set predictions typically
were from the real salaries). The range shown to the user is the prediction
± 0.6 × that residual standard deviation, rounded to the nearest hundred.
This is disclosed here and in the code so it is never mistaken for something
more precise than it is.

## 10. How Personalized Top Factors Are Calculated

The "Top factors affecting your salary" section is **not** a fixed,
one-size-fits-all list — it is recalculated for every profile submitted.

**Method — "swap to typical, and measure the difference":**

1. `train_model.py` records a "typical profile" from the training data: the
   *average* value for each numeric feature (age, experience, hours/week)
   and the *most common category* for each categorical feature (education,
   occupation, gender, country, industry).
2. When you submit your profile, the backend first predicts your salary
   normally.
3. Then, one feature at a time, it takes a copy of *your* profile and
   replaces just that one field with the "typical" value — everything else
   stays as you entered it — and runs that modified profile through the
   same trained pipeline.
4. The difference between your real prediction and this "what if this one
   field were typical" prediction is that feature's dollar impact for you,
   specifically.

This is why changing any field in the form changes the factors, their
order, and their dollar amounts. It is a simplified, easy-to-explain form of
feature attribution (in the same family as permutation/occlusion-based
explanation methods) — not an exact game-theoretic decomposition like SHAP,
but transparent, real, and computed fresh from the trained model every time.

## 11. Installation Instructions

**Requirements:** Python 3.9+ installed on your machine.

```bash
# 1. Move into the project folder
cd Employee-Salary-Prediction

# 2. (Recommended) create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 12. How to Run Locally (VS Code steps)

1. Open the `Employee-Salary-Prediction` folder in VS Code (`File → Open Folder`).
2. Open a terminal inside VS Code (`` Ctrl+` ``).
3. Create and activate a virtual environment (see Section 10 above).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. **Generate the dataset** (only needed once, or whenever you want fresh data):
   ```bash
   python generate_dataset.py
   ```
6. **Train the model** (creates `best_model.pkl` and `model_metrics.json`):
   ```bash
   python train_model.py
   ```
   You should see the data summary, preprocessing log, model comparison
   table, and "BEST MODEL SELECTED" printed in the terminal.
7. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```
8. Open your browser and go to:
   ```
   http://127.0.0.1:8000
   ```
   The full app (form, prediction, what-if simulator, feature importance)
   will load there.
9. (Optional) Explore the auto-generated API docs at:
   ```
   http://127.0.0.1:8000/docs
   ```
10. (Optional) Run the automated test suite:
    ```bash
    pytest -v
    ```
    This runs `tests/test_dataset.py`, `tests/test_preprocessing.py`,
    `tests/test_prediction.py`, and `tests/test_api.py` — covering the
    dataset schema, the cleaning/preprocessing logic, the trained pipeline's
    predictions, and every API endpoint (including a regression test that
    specifically re-creates the PDF-report bug with an `&` in the industry
    name, to make sure it stays fixed).

## 13. Project Structure

Everything lives in a single, flat folder — no nested subfolders:

```
Employee-Salary-Prediction/
│
├── generate_dataset.py     # Creates the realistic synthetic dataset
├── dataset.csv             # The generated training data
├── train_model.py          # Full ML pipeline: load → clean → train → evaluate → save
├── best_model.pkl          # Saved best pipeline (preprocessing + model)
├── model_metrics.json      # Saved metrics, feature importance, etc.
│
├── main.py                 # FastAPI backend (all API endpoints)
├── index.html              # Single-page frontend
├── style.css               # Styling
├── script.js                # Frontend logic (fetch calls, validation, rendering, chart, saved profiles)
│
├── tests/                  # Automated test suite (pytest)
│   ├── test_dataset.py         # Dataset schema, ranges, category coverage
│   ├── test_preprocessing.py   # Cleaning logic + ColumnTransformer shape
│   ├── test_prediction.py      # Trained pipeline + main.py helper functions
│   └── test_api.py             # End-to-end FastAPI endpoint tests (incl. PDF bug regression test)
│
├── requirements.txt
└── README.md
```

`main.py` serves `index.html`, `style.css`, and `script.js` directly, so no
separate `/static` or `/templates` folders are needed — the whole project
stays in one place, as requested.

## 14. Screenshots

_Add screenshots of the running app here after you run it locally:_

- `screenshot-form.png` — the employee information form
- `screenshot-results.png` — prediction, range, and career insights
- `screenshot-whatif.png` — the what-if simulator in action
- `screenshot-importance.png` — feature importance bars
- `screenshot-pdf-report.png` — the downloaded PDF report

## 15. Troubleshooting

**Error on startup: `InconsistentVersionWarning` or `... is not a known BitGenerator`**

This means `best_model.pkl` was saved using a different scikit-learn/numpy
version than the one currently installed. Pickled models are tied to the
library version that created them. Fix it by simply retraining in your own
environment — this regenerates `best_model.pkl` using your installed
versions:

```bash
python generate_dataset.py
python train_model.py
```

Then start the server again with `uvicorn main:app --reload`.

**PDF report download fails / button does nothing / 500 error on `/generate-report`**

Two real, separate bugs in earlier versions of this project could both cause
this, and both are now fixed:

1. **Root cause (the likely actual bug you hit):** `compute_personalized_factors()`
   builds a copy of your profile and swaps in the "typical" reference value
   for one feature at a time, using pandas' `.at[]`. `age` is an integer in
   the form, so its column in the DataFrame was `int64` — but the "typical"
   reference value for age (e.g. `37.49`, an average) is a **float**. On
   modern pandas (2.x/3.x), writing a float into an `int64` column via
   `.at[]`/`.loc[]` raises `TypeError: Invalid value '37.49' for dtype
   'int64'` instead of silently upcasting like older pandas did. This hits
   `/generate-report` **and** `/feature-importance` on any fresh
   `pip install` that pulls in a recent pandas. Fixed by explicitly casting
   all numeric feature columns to `float64` in `employee_to_dataframe()`.
2. `reportlab`'s `Paragraph` treats its input text as a small XML-like
   markup language, so any profile field containing `&`, `<`, or `>` (for
   example the industry "Media & Entertainment") made PDF generation crash.
   Fixed by escaping every dynamic value with `xml.sax.saxutils.escape()`
   before it reaches a `Paragraph`.

Both are covered by dedicated regression tests in `tests/test_prediction.py`
and `tests/test_api.py` so they can't silently come back. If you still see
this error after pulling the latest `main.py`, check the terminal running
`uvicorn` — the fix also adds proper logging, so the real underlying error
will now be printed there instead of just a generic 500.

## 16. Completed in This Update

The items below were originally listed as "Future Improvements" and are now
implemented:

- ✅ **More occupations, countries, and industries** — expanded from 10/6/6
  to 25/16/14. `generate_dataset.py`, `main.py`'s `VALID_*` lists, and the
  `<select>` options in `index.html` were all updated together, and the
  model was retrained on the larger dataset.
- ✅ **Save and compare multiple profiles** — implemented with the browser's
  `localStorage` (no server-side database needed, per the "lightweight"
  option mentioned). Saved profiles persist across page reloads on the same
  browser/device; a "Compare selected" button builds a side-by-side table.
- ✅ **Unit tests** — `tests/test_dataset.py`, `tests/test_preprocessing.py`,
  `tests/test_prediction.py`, and `tests/test_api.py`, covering the dataset,
  the cleaning/preprocessing logic, the trained pipeline, and every API
  endpoint.

## 17. Remaining Future Improvement: Real-World Dataset

**Replacing the synthetic dataset with a real-world salary survey dataset**
is intentionally left as a manual step rather than done automatically here,
for an honest reason: no public dataset was verified to have this project's
exact schema (age, education level, occupation, years of experience, hours
per week, gender, country, industry → salary), and swapping in a
mismatched one silently would make the "realistic, explainable patterns"
claim in `generate_dataset.py` untrue.

If you want to do this yourself, a reasonable path is:

1. Look at Kaggle datasets such as *"Salary Prediction of Data Professions"*,
   *"Data Science Job Salaries"*, or a national salary-survey release (e.g.
   from a government statistics office) — check the columns first.
2. Rename/map its columns to match `FEATURE_COLUMNS` in `train_model.py`
   (`age`, `years_of_experience`, `hours_per_week`, `education_level`,
   `occupation`, `gender`, `country`, `industry`, `salary`). If a column is
   missing (many real datasets don't have `hours_per_week`, for example),
   either drop it from `NUMERIC_FEATURES`/`main.py`'s `EmployeeData` model
   consistently everywhere, or impute a reasonable constant.
3. Update `VALID_OCCUPATIONS` / `VALID_COUNTRIES` / `VALID_INDUSTRIES` in
   `main.py` and the `<select>` options in `index.html` to match the real
   dataset's actual category values (they won't match the synthetic list).
4. Save the new file as `dataset.csv` (same filename, same shape) and run
   `python train_model.py` — everything downstream (API, frontend, PDF
   report, tests) keeps working unchanged as long as the column names match.
5. Re-run `pytest` — `tests/test_dataset.py` will immediately tell you if
   the new file's schema doesn't match what the rest of the project expects.
# EMP_SAL_PRED
