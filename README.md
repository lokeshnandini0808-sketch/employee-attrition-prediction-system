# Employee Attrition Prediction System

## Project Overview

An AI-powered Employee Attrition Prediction System built using Python, Machine Learning, and Streamlit.

The application analyzes employee-related information and estimates attrition risk to support workforce analytics and employee retention planning.

## Key Features

- Professional HR Analytics Dashboard
- Employee Attrition Risk Prediction
- AI Risk Meter
- Explainable AI Insights
- What-If Scenario Analysis
- HR Retention Recommendations
- Employee Data Visualization
- Machine Learning Model Comparison

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Machine Learning Models

The project compares two classification algorithms:

1. Logistic Regression
2. Random Forest Classifier

Logistic Regression was selected for the current prediction system because it achieved a higher attrition-class F1-score.

## Dataset

IBM HR Analytics Employee Attrition & Performance Dataset.

The dataset contains 1,470 fictional employee records and includes attributes such as age, monthly income, job satisfaction, overtime, department, and years at company.

## How to Run

**1. Install required libraries**

```bash
pip install -r requirements.txt
```

**2. Launch the Streamlit application**

```bash
streamlit run app.py
```

## Project Objective

To analyze employee attrition patterns, estimate employee attrition risk, explore hypothetical scenarios, and provide HR-related recommendations.

## Limitations

This project is an educational prototype. Model scores are not calibrated probabilities, and predictions should not be used to make real employment decisions.

## Project Category

Data Science and Artificial Intelligence / Machine Learning

**Problem Statement:** PS-05 — Employee Attrition Analysis & Prediction

**Project Type:** College Internal Hackathon