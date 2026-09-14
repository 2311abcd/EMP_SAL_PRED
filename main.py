"""
main.py
--------
FastAPI backend for the Employee Salary Prediction & Career Insights System.

This file:
    1. Loads the trained pipeline (best_model.pkl) that was created by train_model.py
    2. Loads model_metrics.json (model name, evaluation scores, feature importance)
    3. Defines Pydantic models for request validation
    4. Exposes the required API endpoints:
         GET  /                  -> serves the frontend page
         POST /predict           -> predicts salary + range + career insights
         POST /what-if           -> runs "what if" scenarios through the SAME model
         GET  /model-info        -> returns model name, metrics, features used
         POST /feature-importance-> returns personalized top factors for one profile
         POST /generate-report   -> builds a downloadable PDF report for one employee
    5. Serves style.css and script.js so the whole project can stay in ONE folder
       (no separate /static or /templates folder needed).

Run with:
    uvicorn main:app --reload
"""

import io
import json
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional

# reportlab is used only to build the downloadable PDF report.
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


# =================================================================
# Load the trained pipeline and metrics ONCE when the server starts
# =================================================================
try:
    model_pipeline = joblib.load("best_model.pkl")
    with open("model_metrics.json", "r") as f:
        model_metrics = json.load(f)
except FileNotFoundError:
    raise RuntimeError(
        "best_model.pkl or model_metrics.json not found. "
        "Please run 'python train_model.py' first to train and save the model."
    )

NUMERIC_FEATURES = model_metrics["numeric_features"]
CATEGORICAL_FEATURES = model_metrics["categorical_features"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RESIDUAL_STD = model_metrics["residual_std"]
REFERENCE_VALUES = model_metrics["reference_values"]

# Allowed categorical values (must match the values seen during training).
# Keeping this here lets us give the user a helpful validation error instead
# of a confusing internal server error if they send an unknown category.
VALID_EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
VALID_OCCUPATIONS = [
    "Software Engineer", "Data Analyst", "Sales Executive", "Marketing Manager",
    "HR Specialist", "Financial Analyst", "Product Manager", "Mechanical Engineer",
    "Customer Support", "Operations Manager"
]
VALID_GENDERS = ["Male", "Female", "Other"]
VALID_COUNTRIES = ["USA", "India", "UK", "Canada", "Germany", "Australia"]
VALID_INDUSTRIES = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Education"]


app = FastAPI(
    title="Employee Salary Prediction & Career Insights System",
    description="Predicts estimated annual salary and provides career insights using a trained ML pipeline.",
    version="1.0.0"
)


# =================================================================
# PYDANTIC MODELS (request validation)
# =================================================================

class EmployeeData(BaseModel):
    """Represents the details of one employee, as entered in the form."""

    age: int = Field(..., ge=18, le=70, description="Age must be between 18 and 70")
    education_level: str = Field(..., description="Highest education level")
    occupation: str = Field(..., description="Current job title / occupation")
    years_of_experience: float = Field(..., ge=0, le=50, description="Years of work experience")
    hours_per_week: float = Field(..., ge=1, le=100, description="Average working hours per week")
    gender: str = Field(..., description="Gender (used for demographic display only)")
    country: str = Field(..., description="Country of employment")
    industry: str = Field(..., description="Industry / job category")

    @field_validator("education_level")
    @classmethod
    def validate_education(cls, value):
        if value not in VALID_EDUCATION_LEVELS:
            raise ValueError(f"education_level must be one of {VALID_EDUCATION_LEVELS}")
        return value

    @field_validator("occupation")
    @classmethod
    def validate_occupation(cls, value):
        if value not in VALID_OCCUPATIONS:
            raise ValueError(f"occupation must be one of {VALID_OCCUPATIONS}")
        return value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value):
        if value not in VALID_GENDERS:
            raise ValueError(f"gender must be one of {VALID_GENDERS}")
        return value

    @field_validator("country")
    @classmethod
    def validate_country(cls, value):
        if value not in VALID_COUNTRIES:
            raise ValueError(f"country must be one of {VALID_COUNTRIES}")
        return value

    @field_validator("industry")
    @classmethod
    def validate_industry(cls, value):
        if value not in VALID_INDUSTRIES:
            raise ValueError(f"industry must be one of {VALID_INDUSTRIES}")
        return value


class WhatIfRequest(BaseModel):
    """Request body for the What-If simulator."""

    employee: EmployeeData
    experience_change: Optional[float] = Field(
        default=2.0, description="Years to ADD to current experience for the simulation"
    )
    hours_change: Optional[float] = Field(
        default=0.0, description="Hours per week to ADD to current hours for the simulation"
    )


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def employee_to_dataframe(employee: EmployeeData) -> pd.DataFrame:
    """
    Converts a validated EmployeeData object into a single-row Pandas DataFrame
    with EXACTLY the same column names and order used during training.
    This is the key step that prevents "feature mismatch" errors.
    """
    row = {
        "age": employee.age,
        "years_of_experience": employee.years_of_experience,
        "hours_per_week": employee.hours_per_week,
        "education_level": employee.education_level,
        "occupation": employee.occupation,
        "gender": employee.gender,
        "country": employee.country,
        "industry": employee.industry,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_salary(employee: EmployeeData) -> float:
    """Runs the trained pipeline on one employee record and returns a float salary."""
    input_df = employee_to_dataframe(employee)
    prediction = model_pipeline.predict(input_df)[0]
    return round(float(prediction), 2)


def compute_personalized_factors(employee: EmployeeData, predicted_salary: float) -> list:
    """
    Explains ONE prediction by showing each feature's real dollar impact for
    THIS specific profile -- not a fixed, one-size-fits-all importance score.

    METHOD ("leave-one-out to a typical value"):
    For each feature, we take the employee's real data, but swap just that
    ONE feature for the "typical" value seen in the training data (its
    average for numeric features, its most common category for categorical
    features). We run this modified profile through the SAME trained
    pipeline used everywhere else, and compare the new prediction to the
    employee's actual predicted salary.

        impact = employee's predicted salary - prediction with this feature "neutralized"

    A large positive impact means "having YOUR value for this feature is
    pushing your salary up compared to a typical profile". A negative impact
    means the opposite. Because this recomputation happens for the exact
    profile submitted, the numbers change automatically whenever the user
    changes their inputs -- this is not a static, hardcoded list.

    This is a simplified, easy-to-explain approximation of feature
    attribution (in the same family as permutation/occlusion-based
    explanations), not an exact game-theoretic decomposition like SHAP.
    """
    factors = []
    base_row = employee_to_dataframe(employee)

    for feature in FEATURE_COLUMNS:
        modified_row = base_row.copy()
        modified_row.at[0, feature] = REFERENCE_VALUES[feature]

        neutral_prediction = float(model_pipeline.predict(modified_row)[0])
        impact = round(predicted_salary - neutral_prediction, 2)

        factors.append({
            "feature": feature,
            "impact": impact,
            "direction": "positive" if impact >= 0 else "negative",
            "your_value": str(getattr(employee, feature)),
            "typical_value": str(REFERENCE_VALUES[feature]),
        })

    # Sort so the factor with the biggest effect (up or down) appears first
    factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
    return factors


def get_salary_range(predicted_salary: float) -> dict:
    """
    Builds an estimated salary range around the point prediction.

    METHOD (clearly documented, not claimed to be statistically perfect):
    We use the residual standard deviation measured on the test set during
    training (how far off the model's predictions typically were from the
    real salaries). We build a range of roughly +/- 0.6 * residual_std
    around the prediction. This is an approximation of a confidence band,
    NOT a formal statistical confidence interval.
    """
    margin = 0.6 * RESIDUAL_STD
    minimum = max(0, round(predicted_salary - margin, -2))  # round to nearest 100
    maximum = round(predicted_salary + margin, -2)
    return {"minimum": minimum, "maximum": maximum}


def get_experience_level(years_of_experience: float) -> str:
    """Simple, transparent rule-based bucketing of experience into a label."""
    if years_of_experience < 2:
        return "Entry-Level"
    elif years_of_experience < 6:
        return "Mid-Level Professional"
    elif years_of_experience < 12:
        return "Senior Professional"
    else:
        return "Expert / Leadership Level"


def get_salary_insight(predicted_salary: float) -> dict:
    """
    Compares the predicted salary to the average salary in the training
    dataset to decide if it's Below Average / Average / Above Average.
    """
    # Average salary of the ORIGINAL training data is stored once at startup
    average_salary = model_metrics.get("average_salary")
    if average_salary is None:
        return {"position": "Unavailable", "message": "Historical average not available."}

    difference_pct = ((predicted_salary - average_salary) / average_salary) * 100

    if difference_pct > 10:
        position = "Above Average"
        message = "Your profile predicts a salary above the typical average in our dataset."
    elif difference_pct < -10:
        position = "Below Average"
        message = "Your profile predicts a salary below the typical average in our dataset."
    else:
        position = "Around Average"
        message = "Your profile predicts a salary close to the typical average in our dataset."

    return {"position": position, "message": message, "average_salary": round(average_salary, 2)}


def generate_pdf_report(
    employee: EmployeeData,
    predicted_salary: float,
    salary_range: dict,
    experience_level: str,
    salary_insight: dict,
    career_insight_text: str,
    personalized_factors: list,
) -> bytes:
    """
    Builds a clean, one-page PDF report summarising a single prediction.

    Uses reportlab's Platypus API (SimpleDocTemplate + Paragraph/Table),
    which lays out content top-to-bottom like a simple document instead of
    drawing text at fixed x/y coordinates. The finished PDF is returned as
    raw bytes so it can be streamed straight back in the API response
    without ever being saved to disk.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#1B4332")
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], textColor=colors.HexColor("#1B4332"),
        spaceBefore=16, spaceAfter=8,
    )
    body_style = ParagraphStyle("Body", parent=styles["Normal"], leading=15)
    muted_style = ParagraphStyle("Muted", parent=styles["Normal"], textColor=colors.grey, fontSize=9)

    story = []

    # ---- Header ----
    story.append(Paragraph("Employee Salary Prediction Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}", muted_style
    ))
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DCE4DF")))
    story.append(Spacer(1, 10))

    # ---- Employee profile table ----
    story.append(Paragraph("Employee Profile", heading_style))
    profile_rows = [
        ["Age", str(employee.age)],
        ["Education Level", employee.education_level],
        ["Occupation", employee.occupation],
        ["Years of Experience", f"{employee.years_of_experience}"],
        ["Hours per Week", f"{employee.hours_per_week}"],
        ["Gender", employee.gender],
        ["Country", employee.country],
        ["Industry", employee.industry],
    ]
    profile_table = Table(profile_rows, colWidths=[6 * cm, 9 * cm])
    profile_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5C6B64")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E5E5E5")),
    ]))
    story.append(profile_table)

    # ---- Prediction results ----
    story.append(Paragraph("Prediction Results", heading_style))
    result_rows = [
        ["Estimated Annual Salary", f"${predicted_salary:,.2f}"],
        ["Estimated Range", f"${salary_range['minimum']:,.2f} - ${salary_range['maximum']:,.2f}"],
        ["Experience Level", experience_level],
        ["Salary Position", salary_insight["position"]],
    ]
    result_table = Table(result_rows, colWidths=[6 * cm, 9 * cm])
    result_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5C6B64")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#1B4332")),
        ("FONTSIZE", (1, 0), (1, 0), 13),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#E5E5E5")),
    ]))
    story.append(result_table)

    # ---- Career insight ----
    story.append(Paragraph("Career Insight", heading_style))
    story.append(Paragraph(career_insight_text, body_style))

    # ---- Feature importance (personalized to THIS profile) ----
    if personalized_factors:
        story.append(Paragraph("Top Factors Affecting Your Salary", heading_style))
        story.append(Paragraph(
            "Calculated specifically for your profile by comparing each field to a "
            "typical employee in the training data.",
            muted_style,
        ))
        story.append(Spacer(1, 6))
        factor_rows = [["#", "Factor", "Your Value", "Impact"]]
        for i, factor in enumerate(personalized_factors, start=1):
            factor_name = factor["feature"].replace("_", " ").title()
            sign = "+" if factor["impact"] >= 0 else "-"
            impact_text = f"{sign}${abs(factor['impact']):,.0f}"
            factor_rows.append([str(i), factor_name, factor["your_value"], impact_text])
        factor_table = Table(factor_rows, colWidths=[1 * cm, 5.5 * cm, 4.5 * cm, 4 * cm])
        factor_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B4332")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAF8")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E5E5")),
            ("ALIGN", (3, 0), (3, -1), "RIGHT"),
        ]))
        story.append(factor_table)

    # ---- Footer / disclaimer ----
    story.append(Spacer(1, 22))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DCE4DF")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This report is generated by a trained machine learning regression model "
        "(Employee Salary Prediction &amp; Career Insights System). The predicted "
        "salary, range, and insights are model-based estimates, not guarantees of "
        "actual pay.",
        muted_style,
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def build_career_insight_text(employee: EmployeeData, experience_level: str, salary_insight: dict) -> str:
    """Creates a short, human-readable insight sentence based on real input values."""
    parts = []
    parts.append(f"You are currently classified as a '{experience_level}' based on "
                 f"{employee.years_of_experience} years of experience.")
    if employee.education_level in ("Master's", "PhD"):
        parts.append("Your advanced education level is a strong positive factor in your predicted salary.")
    if salary_insight["position"] == "Above Average":
        parts.append("Your experience, education and industry combine to place you above the typical average.")
    elif salary_insight["position"] == "Below Average":
        parts.append("Gaining more experience or an additional qualification could help move you closer to the average.")
    else:
        parts.append("Your predicted salary sits close to the typical average for similar profiles.")
    return " ".join(parts)


# =================================================================
# ROUTES
# =================================================================

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the main single-page frontend."""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/style.css")
def serve_css():
    return FileResponse("style.css", media_type="text/css")


@app.get("/script.js")
def serve_js():
    return FileResponse("script.js", media_type="application/javascript")


@app.post("/predict")
def predict(employee: EmployeeData):
    """
    Predicts the estimated annual salary for one employee and
    returns the prediction, an estimated range, and career insights.
    """
    try:
        predicted_salary = predict_salary(employee)
        salary_range = get_salary_range(predicted_salary)
        experience_level = get_experience_level(employee.years_of_experience)
        salary_insight = get_salary_insight(predicted_salary)
        career_insight_text = build_career_insight_text(employee, experience_level, salary_insight)

        return {
            "predicted_salary": predicted_salary,
            "salary_range": salary_range,
            "experience_level": experience_level,
            "salary_insight": salary_insight["position"],
            "salary_insight_message": salary_insight["message"],
            "career_insight": career_insight_text,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(error)}")


@app.post("/generate-report")
def generate_report(employee: EmployeeData):
    """
    Builds a downloadable PDF report for one employee.

    This does NOT trust any prediction numbers from the client — it
    recomputes the prediction, range, and insights itself using the same
    trained pipeline as /predict, so the PDF always matches what the
    model actually produces. Nothing is saved to disk on the server;
    the PDF is built in memory and streamed straight back in the
    response.
    """
    try:
        predicted_salary = predict_salary(employee)
        salary_range = get_salary_range(predicted_salary)
        experience_level = get_experience_level(employee.years_of_experience)
        salary_insight = get_salary_insight(predicted_salary)
        career_insight_text = build_career_insight_text(employee, experience_level, salary_insight)
        personalized_factors = compute_personalized_factors(employee, predicted_salary)

        pdf_bytes = generate_pdf_report(
            employee, predicted_salary, salary_range,
            experience_level, salary_insight, career_insight_text,
            personalized_factors,
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=salary_prediction_report.pdf"},
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(error)}")


@app.post("/what-if")
def what_if(request: WhatIfRequest):
    """
    Runs a 'What-If' simulation.

    IMPORTANT: This does NOT invent fake numbers. It creates a modified COPY
    of the user's real input (e.g., +2 years of experience) and passes that
    copy through the exact same trained pipeline used in /predict. The
    difference between the two real predictions is the "potential increase".
    """
    try:
        employee = request.employee

        current_salary = predict_salary(employee)

        # Create a modified copy of the employee's data for the scenario
        modified_data = employee.model_copy(deep=True)
        modified_data.years_of_experience = max(
            0, employee.years_of_experience + request.experience_change
        )
        modified_data.hours_per_week = max(
            1, min(100, employee.hours_per_week + request.hours_change)
        )

        new_salary = predict_salary(modified_data)
        difference = round(new_salary - current_salary, 2)

        explanation_parts = []
        if request.experience_change:
            explanation_parts.append(
                f"increasing experience by {request.experience_change} year(s)"
            )
        if request.hours_change:
            explanation_parts.append(
                f"changing working hours by {request.hours_change} hour(s) per week"
            )
        explanation = " and ".join(explanation_parts) if explanation_parts else "no change"

        suggestion = (
            f"Model-based estimate: {explanation} could change the estimated salary "
            f"by approximately ${abs(difference):,.2f} "
            f"({'increase' if difference >= 0 else 'decrease'}). "
            f"This is an estimate from the trained model, not a guarantee."
        )

        return {
            "current_salary": current_salary,
            "new_salary": new_salary,
            "difference": difference,
            "scenario": f"Scenario: {explanation}",
            "explanation": suggestion,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"What-If simulation failed: {str(error)}")


@app.get("/model-info")
def model_info():
    """Returns the best model's name, its evaluation metrics, and features used."""
    return {
        "best_model_name": model_metrics["best_model_name"],
        "model_metrics": model_metrics["all_model_results"],
        "features_used": FEATURE_COLUMNS,
        "training_rows": model_metrics["training_rows"],
        "testing_rows": model_metrics["testing_rows"],
    }


@app.post("/feature-importance")
def feature_importance(employee: EmployeeData):
    """
    Returns the top factors affecting THIS employee's predicted salary,
    computed fresh for their specific profile (see compute_personalized_factors
    for the method). Every field in the request changes these numbers.
    """
    try:
        predicted_salary = predict_salary(employee)
        factors = compute_personalized_factors(employee, predicted_salary)
        return {
            "supported": True,
            "best_model_name": model_metrics["best_model_name"],
            "predicted_salary": predicted_salary,
            "top_factors": factors,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(error)}")
