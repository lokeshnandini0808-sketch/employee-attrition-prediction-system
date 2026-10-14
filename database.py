"""
Database module for AI-Powered Employee Management & Attrition Prediction System.
Handles SQLite connection, schema creation, CRUD operations, and seeding.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

DB_PATH = Path(__file__).resolve().parent / "hr_attrition.db"

# 30 model features required by attrition_model_v2.pkl
FEATURE_COLUMNS = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
]

def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create employees table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        Age INTEGER,
        BusinessTravel TEXT,
        DailyRate INTEGER,
        Department TEXT,
        DistanceFromHome INTEGER,
        Education INTEGER,
        EducationField TEXT,
        EnvironmentSatisfaction INTEGER,
        Gender TEXT,
        HourlyRate INTEGER,
        JobInvolvement INTEGER,
        JobLevel INTEGER,
        JobRole TEXT,
        JobSatisfaction INTEGER,
        MaritalStatus TEXT,
        MonthlyIncome REAL,
        MonthlyRate INTEGER,
        NumCompaniesWorked INTEGER,
        OverTime TEXT,
        PercentSalaryHike INTEGER,
        PerformanceRating INTEGER,
        RelationshipSatisfaction INTEGER,
        StockOptionLevel INTEGER,
        TotalWorkingYears INTEGER,
        TrainingTimesLastYear INTEGER,
        WorkLifeBalance INTEGER,
        YearsAtCompany INTEGER,
        YearsInCurrentRole INTEGER,
        YearsSinceLastPromotion INTEGER,
        YearsWithCurrManager INTEGER,
        predicted_risk TEXT DEFAULT 'Unassessed',
        attrition_score REAL DEFAULT NULL,
        prediction_label TEXT DEFAULT NULL,
        last_prediction_date TEXT DEFAULT NULL
    )
    """)

    # Create prediction history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT NOT NULL,
        name TEXT NOT NULL,
        attrition_score REAL NOT NULL,
        risk_level TEXT NOT NULL,
        prediction_label TEXT NOT NULL,
        prediction_date TEXT NOT NULL,
        key_factors TEXT DEFAULT '[]'
    )
    """)

    conn.commit()
    conn.close()

def get_next_emp_id():
    """Generate sequential employee ID like EMP-1001, EMP-1002, etc."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT emp_id FROM employees ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    max_num = 1000
    for row in rows:
        val = row["emp_id"]
        if val and val.startswith("EMP-"):
            try:
                num = int(val.replace("EMP-", ""))
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return f"EMP-{max_num + 1}"

def add_employee(data):
    """Insert a new employee record."""
    conn = get_connection()
    cursor = conn.cursor()

    if "created_at" not in data or not data["created_at"]:
        data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cols = ["emp_id", "name", "created_at"] + [c for c in FEATURE_COLUMNS if c in data]
    optional_pred = ["predicted_risk", "attrition_score", "prediction_label", "last_prediction_date"]
    for op in optional_pred:
        if op in data and data[op] is not None:
            cols.append(op)

    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    values = [data.get(c) for c in cols]

    cursor.execute(f"INSERT INTO employees ({col_str}) VALUES ({placeholders})", values)
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def update_employee(emp_id, data):
    """Update an existing employee record."""
    conn = get_connection()
    cursor = conn.cursor()

    cols_to_update = []
    values = []
    for k, v in data.items():
        if k in FEATURE_COLUMNS or k in ["name", "predicted_risk", "attrition_score", "prediction_label", "last_prediction_date"]:
            cols_to_update.append(f"{k} = ?")
            values.append(v)

    if not cols_to_update:
        conn.close()
        return False

    values.append(emp_id)
    sql = f"UPDATE employees SET {', '.join(cols_to_update)} WHERE emp_id = ?"
    cursor.execute(sql, values)
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

def delete_employee(emp_id):
    """Delete an employee and their prediction records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM employees WHERE emp_id = ?", (emp_id,))
    cursor.execute("DELETE FROM prediction_history WHERE emp_id = ?", (emp_id,))
    conn.commit()
    conn.close()
    return True

def get_employee(emp_id):
    """Retrieve a single employee as a dictionary."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_employees():
    """Retrieve all employees as a Pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM employees ORDER BY id DESC", conn)
    conn.close()
    return df

def update_prediction(emp_id, attrition_score, risk_level, prediction_label, prediction_date):
    """Update prediction results for a given employee in employees table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE employees
        SET attrition_score = ?,
            predicted_risk = ?,
            prediction_label = ?,
            last_prediction_date = ?
        WHERE emp_id = ?
    """, (attrition_score, risk_level, prediction_label, prediction_date, emp_id))
    conn.commit()
    conn.close()

def log_prediction_history(emp_id, name, attrition_score, risk_level, prediction_label, prediction_date, key_factors=None):
    """Log a prediction event to prediction_history table."""
    conn = get_connection()
    cursor = conn.cursor()
    factors_json = json.dumps(key_factors) if key_factors else "[]"
    cursor.execute("""
        INSERT INTO prediction_history (emp_id, name, attrition_score, risk_level, prediction_label, prediction_date, key_factors)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (emp_id, name, attrition_score, risk_level, prediction_label, prediction_date, factors_json))
    conn.commit()
    conn.close()

def get_prediction_history(limit=50):
    """Fetch recent prediction history records."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,)
    )
    conn.close()
    return df

def seed_sample_employees():
    """Seed initial realistic employee records if database is empty."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM employees")
    cnt = cursor.fetchone()["cnt"]
    conn.close()

    if cnt > 0:
        return False  # Already seeded or has data

    demo_employees = [
        {
            "emp_id": "EMP-1001",
            "name": "Sarah Jenkins",
            "created_at": "2026-09-15 09:30:00",
            "Age": 31,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 750,
            "Department": "Research & Development",
            "DistanceFromHome": 4,
            "Education": 3,
            "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 3,
            "Gender": "Female",
            "HourlyRate": 72,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobRole": "Research Scientist",
            "JobSatisfaction": 4,
            "MaritalStatus": "Married",
            "MonthlyIncome": 5200.0,
            "MonthlyRate": 15200,
            "NumCompaniesWorked": 1,
            "OverTime": "No",
            "PercentSalaryHike": 15,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 4,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 9,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 7,
            "YearsInCurrentRole": 5,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 5,
            "predicted_risk": "Low",
            "attrition_score": 0.14,
            "prediction_label": "Retained / Low Risk",
            "last_prediction_date": "2026-10-12 14:15:00"
        },
        {
            "emp_id": "EMP-1002",
            "name": "Marcus Vance",
            "created_at": "2026-09-18 11:20:00",
            "Age": 28,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 480,
            "Department": "Sales",
            "DistanceFromHome": 24,
            "Education": 2,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 55,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": "Sales Representative",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2850.0,
            "MonthlyRate": 19400,
            "NumCompaniesWorked": 4,
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 5,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 2,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 1,
            "predicted_risk": "High",
            "attrition_score": 0.88,
            "prediction_label": "At Risk",
            "last_prediction_date": "2026-10-13 10:00:00"
        },
        {
            "emp_id": "EMP-1003",
            "name": "Dr. Priya Sharma",
            "created_at": "2026-09-22 14:45:00",
            "Age": 44,
            "BusinessTravel": "Non-Travel",
            "DailyRate": 1120,
            "Department": "Research & Development",
            "DistanceFromHome": 6,
            "Education": 4,
            "EducationField": "Medical",
            "EnvironmentSatisfaction": 4,
            "Gender": "Female",
            "HourlyRate": 85,
            "JobInvolvement": 4,
            "JobLevel": 4,
            "JobRole": "Healthcare Representative",
            "JobSatisfaction": 3,
            "MaritalStatus": "Married",
            "MonthlyIncome": 12800.0,
            "MonthlyRate": 21000,
            "NumCompaniesWorked": 2,
            "OverTime": "No",
            "PercentSalaryHike": 18,
            "PerformanceRating": 4,
            "RelationshipSatisfaction": 4,
            "StockOptionLevel": 2,
            "TotalWorkingYears": 22,
            "TrainingTimesLastYear": 4,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 14,
            "YearsInCurrentRole": 8,
            "YearsSinceLastPromotion": 4,
            "YearsWithCurrManager": 7,
            "predicted_risk": "Low",
            "attrition_score": 0.11,
            "prediction_label": "Retained / Low Risk",
            "last_prediction_date": "2026-10-14 09:30:00"
        },
        {
            "emp_id": "EMP-1004",
            "name": "David Brooks",
            "created_at": "2026-09-29 16:10:00",
            "Age": 35,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 690,
            "Department": "Human Resources",
            "DistanceFromHome": 16,
            "Education": 3,
            "EducationField": "Human Resources",
            "EnvironmentSatisfaction": 2,
            "Gender": "Male",
            "HourlyRate": 64,
            "JobInvolvement": 2,
            "JobLevel": 2,
            "JobRole": "Human Resources",
            "JobSatisfaction": 2,
            "MaritalStatus": "Single",
            "MonthlyIncome": 4400.0,
            "MonthlyRate": 13500,
            "NumCompaniesWorked": 3,
            "OverTime": "Yes",
            "PercentSalaryHike": 13,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 10,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 4,
            "YearsInCurrentRole": 2,
            "YearsSinceLastPromotion": 3,
            "YearsWithCurrManager": 2,
            "predicted_risk": "Medium",
            "attrition_score": 0.49,
            "prediction_label": "At Risk",
            "last_prediction_date": "2026-10-14 11:20:00"
        },
        {
            "emp_id": "EMP-1005",
            "name": "Elena Rostova",
            "created_at": "2026-10-02 10:15:00",
            "Age": 38,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 920,
            "Department": "Sales",
            "DistanceFromHome": 8,
            "Education": 3,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 3,
            "Gender": "Female",
            "HourlyRate": 68,
            "JobInvolvement": 3,
            "JobLevel": 3,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 3,
            "MaritalStatus": "Married",
            "MonthlyIncome": 7800.0,
            "MonthlyRate": 16700,
            "NumCompaniesWorked": 1,
            "OverTime": "No",
            "PercentSalaryHike": 16,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 14,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 9,
            "YearsInCurrentRole": 6,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 6,
            "predicted_risk": "Low",
            "attrition_score": 0.18,
            "prediction_label": "Retained / Low Risk",
            "last_prediction_date": "2026-10-14 12:45:00"
        },
        {
            "emp_id": "EMP-1006",
            "name": "Kevin Patel",
            "created_at": "2026-10-08 13:00:00",
            "Age": 26,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 530,
            "Department": "Research & Development",
            "DistanceFromHome": 21,
            "Education": 2,
            "EducationField": "Technical Degree",
            "EnvironmentSatisfaction": 2,
            "Gender": "Male",
            "HourlyRate": 48,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": "Laboratory Technician",
            "JobSatisfaction": 2,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2980.0,
            "MonthlyRate": 18200,
            "NumCompaniesWorked": 2,
            "OverTime": "Yes",
            "PercentSalaryHike": 12,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 4,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 3,
            "YearsInCurrentRole": 2,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 1,
            "predicted_risk": "High",
            "attrition_score": 0.79,
            "prediction_label": "At Risk",
            "last_prediction_date": "2026-10-14 15:30:00"
        }
    ]

    for emp in demo_employees:
        add_employee(emp)
        if emp.get("attrition_score") is not None:
            log_prediction_history(
                emp["emp_id"],
                emp["name"],
                emp["attrition_score"],
                emp["predicted_risk"],
                emp["prediction_label"],
                emp["last_prediction_date"],
                ["Initial Demonstration Assessment"]
            )
    return True
