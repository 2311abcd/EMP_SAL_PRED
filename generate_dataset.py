"""
generate_dataset.py
--------------------
This script creates the dataset used to train our Salary Prediction model.

WHY A GENERATED DATASET?
-------------------------
Most free public salary datasets (like the famous "UCI Adult Income" dataset)
only tell you whether a person earns "<=50K" or ">50K" per year. That is a
CLASSIFICATION problem, not a REGRESSION problem, so it cannot give an actual
predicted salary number like "$72,500".

Since this project specifically asks for real salary REGRESSION (an actual
dollar amount, a range, and a What-If simulator), we generate a synthetic but
REALISTIC dataset. The salary for every employee is calculated using a formula
that mimics how salaries actually work in the real world:

    salary = f(education, occupation, experience, hours worked, country, industry) + random noise

This is a common and honest technique used in ML learning projects when a
free regression-ready dataset with the exact fields you need does not exist.
The relationships below (more experience = more pay, PhD > Bachelor's, tech
industry pays more, etc.) are realistic, so the trained model will show
genuine, explainable patterns during training and prediction.

NOTE: Gender is included ONLY as a demographic field for the user to fill in.
It intentionally has NO effect on the generated salary, so our model never
learns a gender-based pay gap. This is a deliberate and responsible design
choice.
"""

import numpy as np
import pandas as pd

# Fix the random seed so the dataset is the same every time we generate it
np.random.seed(42)

NUM_RECORDS = 12000

# ---------------------------------------------------------------
# Step 1: Define the possible categories for each categorical column
# ---------------------------------------------------------------
# NOTE ON COVERAGE (Sep 2026 update): expanded from the original 10
# occupations / 6 countries / 6 industries to a much broader set so the
# model and the What-If simulator generalize to more real-world profiles.
education_levels = ["High School", "Bachelor's", "Master's", "PhD"]

occupations = [
    "Software Engineer", "Data Analyst", "Data Scientist", "DevOps Engineer",
    "Sales Executive", "Marketing Manager", "HR Specialist", "Financial Analyst",
    "Accountant", "Product Manager", "Project Manager", "Business Analyst",
    "Mechanical Engineer", "Civil Engineer", "Electrical Engineer",
    "Customer Support", "Operations Manager", "UX Designer", "Graphic Designer",
    "Content Writer", "Legal Counsel", "Research Scientist", "Registered Nurse",
    "Teacher", "Network Administrator",
]

genders = ["Male", "Female", "Other"]

countries = [
    "USA", "India", "UK", "Canada", "Germany", "Australia",
    "France", "Netherlands", "Ireland", "Singapore", "UAE", "Japan",
    "Brazil", "South Africa", "New Zealand", "Mexico",
]

industries = [
    "Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Education",
    "Telecommunications", "Energy", "Government", "Media & Entertainment",
    "Real Estate", "Hospitality", "Automotive", "Pharmaceuticals",
]

# ---------------------------------------------------------------
# Step 2: Define base salary effect for each category
# (This is what makes the dataset "realistic" instead of random)
# ---------------------------------------------------------------
education_multiplier = {"High School": 1.0, "Bachelor's": 1.3, "Master's": 1.55, "PhD": 1.8}

occupation_base_pay = {
    "Software Engineer": 45000, "Data Analyst": 38000, "Data Scientist": 50000,
    "DevOps Engineer": 47000, "Sales Executive": 32000, "Marketing Manager": 40000,
    "HR Specialist": 30000, "Financial Analyst": 42000, "Accountant": 34000,
    "Product Manager": 48000, "Project Manager": 41000, "Business Analyst": 37000,
    "Mechanical Engineer": 36000, "Civil Engineer": 35000, "Electrical Engineer": 37000,
    "Customer Support": 26000, "Operations Manager": 34000, "UX Designer": 39000,
    "Graphic Designer": 30000, "Content Writer": 27000, "Legal Counsel": 55000,
    "Research Scientist": 46000, "Registered Nurse": 33000, "Teacher": 28000,
    "Network Administrator": 33000,
}

country_multiplier = {
    "USA": 1.5, "UK": 1.3, "Canada": 1.25, "Australia": 1.3, "Germany": 1.2,
    "India": 0.5, "France": 1.15, "Netherlands": 1.28, "Ireland": 1.22,
    "Singapore": 1.35, "UAE": 1.18, "Japan": 1.1, "Brazil": 0.55,
    "South Africa": 0.45, "New Zealand": 1.2, "Mexico": 0.5,
}

industry_multiplier = {
    "Technology": 1.3, "Finance": 1.25, "Healthcare": 1.15, "Retail": 0.9,
    "Manufacturing": 1.0, "Education": 0.85, "Telecommunications": 1.12,
    "Energy": 1.2, "Government": 0.95, "Media & Entertainment": 1.05,
    "Real Estate": 1.08, "Hospitality": 0.88, "Automotive": 1.02,
    "Pharmaceuticals": 1.22,
}

# ---------------------------------------------------------------
# Step 3: Randomly generate employee records
# ---------------------------------------------------------------
rows = []
for _ in range(NUM_RECORDS):
    age = int(np.clip(np.random.normal(38, 10), 18, 65))
    education = np.random.choice(education_levels, p=[0.25, 0.40, 0.25, 0.10])
    occupation = np.random.choice(occupations)
    # Experience roughly tied to age so the data feels realistic
    max_possible_experience = max(age - 18, 0)
    years_experience = round(np.clip(np.random.normal(max_possible_experience * 0.5, 4), 0, max_possible_experience), 1)
    hours_per_week = round(np.clip(np.random.normal(42, 8), 20, 70), 1)
    gender = np.random.choice(genders, p=[0.48, 0.48, 0.04])
    country = np.random.choice(countries)
    industry = np.random.choice(industries)

    # ---- Salary formula (the "ground truth" relationship) ----
    base = 30000
    exp_effect = 2500 * np.sqrt(years_experience)          # diminishing returns on experience
    hours_effect = (hours_per_week - 40) * 300              # more hours -> more pay
    raw_salary = (base + occupation_base_pay[occupation] + exp_effect + hours_effect)
    raw_salary *= education_multiplier[education]
    raw_salary *= country_multiplier[country]
    raw_salary *= industry_multiplier[industry]

    # Add random real-world noise (+/- ~8%)
    noise = np.random.normal(0, 0.08 * raw_salary)
    salary = raw_salary + noise
    salary = float(np.clip(salary, 15000, 300000))

    rows.append({
        "age": age,
        "education_level": education,
        "occupation": occupation,
        "years_of_experience": years_experience,
        "hours_per_week": hours_per_week,
        "gender": gender,
        "country": country,
        "industry": industry,
        "salary": round(salary, 2)
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------
# Step 4: Deliberately inject a few missing values and duplicates
# so that train_model.py has real preprocessing work to do
# (this mimics a messy real-world dataset)
# ---------------------------------------------------------------
missing_idx = np.random.choice(df.index, size=80, replace=False)
df.loc[missing_idx, "hours_per_week"] = np.nan

duplicate_rows = df.sample(30, random_state=1)
df = pd.concat([df, duplicate_rows], ignore_index=True)

df.to_csv("dataset.csv", index=False)
print(f"dataset.csv created with {len(df)} rows and {len(df.columns)} columns.")
