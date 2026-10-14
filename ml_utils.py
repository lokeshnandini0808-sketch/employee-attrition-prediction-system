"""
Machine Learning utility module for Attrition Prediction.
Handles loading attrition_model_v2.pkl, matching features, data validation,
generating attrition probability, risk categories, and explainability insights.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "attrition_model_v2.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
DATASET_PATH = BASE_DIR / "employee_data.csv"

# Predefined categorical options from the IBM dataset
CATEGORICAL_OPTIONS = {
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "JobRole": [
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative"
    ],
    "OverTime": ["No", "Yes"],
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "EducationField": [
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree"
    ],
    "Gender": ["Female", "Male"],
    "MaritalStatus": ["Divorced", "Married", "Single"]
}

# Baseline dataset defaults (medians/modes)
DATASET_DEFAULTS = {
    "Age": 36,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 802,
    "Department": "Research & Development",
    "DistanceFromHome": 7,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 66,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 3,
    "MaritalStatus": "Married",
    "MonthlyIncome": 4919.0,
    "MonthlyRate": 14235,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 14,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
}

_cached_model = None
_cached_features = None

def get_model_and_features():
    """Load model pipeline and feature names with caching."""
    global _cached_model, _cached_features
    if _cached_model is None or _cached_features is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"Feature definitions not found at {FEATURES_PATH}")
        _cached_model = joblib.load(MODEL_PATH)
        _cached_features = joblib.load(FEATURES_PATH)
    return _cached_model, _cached_features

def prepare_input_dataframe(emp_data, features):
    """
    Constructs a 1-row DataFrame matching the exact 30 model features
    with correct numerical and categorical types.
    Ensures employee name and ID are stripped out.
    """
    row = {}
    for f in features:
        val = emp_data.get(f, DATASET_DEFAULTS.get(f))
        if val is None:
            val = DATASET_DEFAULTS.get(f)

        # Cast to proper dtype
        if f in [
            "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
            "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyRate",
            "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
            "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
            "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
            "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
        ]:
            row[f] = int(round(float(val)))
        elif f == "MonthlyIncome":
            row[f] = float(val)
        else:
            row[f] = str(val)

    # Reorder columns exactly as expected during training
    input_df = pd.DataFrame([row], columns=features)
    return input_df

def predict_attrition(emp_data):
    """
    Run prediction using attrition_model_v2.pkl.
    Returns a dict with:
    - attrition_score (float, 0.0 - 1.0)
    - classification (str, "At Risk" or "Retained / Low Risk")
    - binary_prediction (int, 1 or 0)
    - risk_category (str, "Low", "Medium", "High")
    - risk_color (str, hex code)
    - risk_drivers (list of str)
    - protective_factors (list of str)
    - recommendations (list of str)
    """
    model, features = get_model_and_features()
    input_df = prepare_input_dataframe(emp_data, features)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    attrition_score = float(probabilities[1])

    # Illustrative Risk Tiers:
    # < 30%: Low, 30% - 60%: Medium, >= 60%: High
    if attrition_score < 0.30:
        risk_category = "Low"
        risk_color = "#10b981"  # Emerald green
    elif attrition_score < 0.60:
        risk_category = "Medium"
        risk_color = "#f59e0b"  # Amber orange
    else:
        risk_category = "High"
        risk_color = "#ef4444"  # Rose red

    classification = "At Risk of Leaving" if prediction == 1 else "Likely to Stay / Retained"

    explain = explain_prediction_factors(emp_data, attrition_score)

    return {
        "attrition_score": attrition_score,
        "classification": classification,
        "binary_prediction": prediction,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "risk_drivers": explain["risk_drivers"],
        "protective_factors": explain["protective_factors"],
        "recommendations": explain["recommendations"]
    }

def explain_prediction_factors(emp_data, attrition_score):
    """
    Analyze employee attributes against model coefficients to generate
    meaningful risk drivers, protective factors, and HR retention recommendations.
    """
    drivers = []
    protective = []
    recommendations = []

    # 1. Overtime Analysis
    ot = str(emp_data.get("OverTime", "No"))
    if ot == "Yes":
        drivers.append("Mandatory or frequent overtime (+76% model log-odds weight)")
        recommendations.append("Review weekly workload and explore shift balancing or overtime compensation caps.")
    else:
        protective.append("Standard work hours without routine overtime")

    # 2. Job Satisfaction
    js = int(emp_data.get("JobSatisfaction", 3))
    if js <= 2:
        drivers.append(f"Low Job Satisfaction rating ({js}/4)")
        recommendations.append("Conduct an informal 1-on-1 check-in to uncover friction points in daily role.")
    elif js >= 3:
        protective.append(f"High Job Satisfaction rating ({js}/4)")

    # 3. Environment Satisfaction
    es = int(emp_data.get("EnvironmentSatisfaction", 3))
    if es <= 2:
        drivers.append(f"Sub-optimal workplace environment rating ({es}/4)")
        recommendations.append("Investigate team culture, physical workplace, or hybrid arrangement support.")
    elif es >= 3:
        protective.append(f"Strong workplace environment contentment ({es}/4)")

    # 4. Work-Life Balance
    wlb = int(emp_data.get("WorkLifeBalance", 3))
    if wlb <= 2:
        drivers.append(f"Constrained Work-Life Balance score ({wlb}/4)")
        recommendations.append("Discuss flexible scheduling, remote work allowances, or stress mitigation.")
    else:
        protective.append(f"Favorable Work-Life Balance score ({wlb}/4)")

    # 5. Career Progression & Promotion
    yslp = int(emp_data.get("YearsSinceLastPromotion", 1))
    yac = int(emp_data.get("YearsAtCompany", 5))
    if yslp >= 4:
        drivers.append(f"Stagnation signal: {yslp} years since last career promotion")
        recommendations.append("Formulate a clear internal growth roadmap or consider title/compensation adjustment.")
    elif yslp <= 1:
        protective.append("Recent promotion within the past 12 months")

    # 6. Commute Distance
    dfh = int(emp_data.get("DistanceFromHome", 7))
    if dfh >= 15:
        drivers.append(f"Lengthy daily commute distance ({dfh} miles/km)")
        recommendations.append("Offer hybrid telecommuting options to reduce daily travel fatigue.")
    else:
        protective.append(f"Manageable proximity to work ({dfh} miles/km)")

    # 7. Travel Frequency
    bt = str(emp_data.get("BusinessTravel", "Travel_Rarely"))
    if bt == "Travel_Frequently":
        drivers.append("High business travel cadence ('Travel_Frequently')")
        recommendations.append("Rotate travel obligations across team members to prevent travel burnout.")
    elif bt == "Non-Travel":
        protective.append("Zero business travel demands")

    # 8. Manager Relationship & Stability
    ywcm = int(emp_data.get("YearsWithCurrManager", 3))
    if ywcm >= 4:
        protective.append(f"Established rapport with current manager ({ywcm} years)")
    elif ywcm <= 1 and yac > 2:
        drivers.append("Recent managerial change or brief relationship with current manager (<1 year)")
        recommendations.append("Encourage regular skip-level check-ins to build managerial alignment.")

    # 9. Role Specific Insights
    role = str(emp_data.get("JobRole", ""))
    if role in ["Laboratory Technician", "Sales Representative"]:
        drivers.append(f"Higher-turnover industry benchmark role: {role}")
    elif role in ["Research Director", "Healthcare Representative", "Manager"]:
        protective.append(f"High-tenure leadership role: {role}")

    # Fallback recommendations if none triggered
    if not recommendations:
        if attrition_score >= 0.5:
            recommendations.append("Schedule a quarterly retention review to maintain high engagement.")
        else:
            recommendations.append("Employee displays strong retention indicators. Continue positive engagement practices.")

    return {
        "risk_drivers": drivers,
        "protective_factors": protective,
        "recommendations": recommendations
    }
