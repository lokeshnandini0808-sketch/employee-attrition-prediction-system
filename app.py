"""
AI-Powered Employee Management & Attrition Prediction System
Built for College Data Science & AI/ML Hackathon 2026.
Stack: Streamlit, SQLite, Scikit-learn, Pandas, NumPy, Plotly.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import io

import database
import ml_utils

# ---------------------------------------------------------
# Page Configuration & Modern SaaS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="TalentPulse AI | HR Management & Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise HR Styling
st.markdown("""
<style>
    /* Global Font & Smoothness */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.25);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 15px;
        opacity: 0.92;
        color: #e0f2fe !important;
        margin-bottom: 0px;
    }
    
    /* Modern Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* Badges */
    .badge-low {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #a7f3d0;
    }
    .badge-medium {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #fde68a;
    }
    .badge-high {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #fecaca;
    }
    .badge-unassessed {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #cbd5e1;
    }

    /* Section Cards */
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    
    /* Custom Sidebar Header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0 16px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .sidebar-brand h2 {
        font-size: 20px;
        font-weight: 800;
        margin: 0;
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Initialize Database & State
# ---------------------------------------------------------
database.init_db()
database.seed_sample_employees()

@st.cache_data
def load_historical_dataset():
    """Load IBM HR Analytics historical dataset."""
    df = pd.read_csv("employee_data.csv")
    return df

try:
    historical_df = load_historical_dataset()
except Exception as e:
    historical_df = pd.DataFrame()
    st.sidebar.warning(f"Note: historical dataset employee_data.csv not loaded: {e}")

# Session state routing helpers
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"
if "selected_emp_id" not in st.session_state:
    st.session_state.selected_emp_id = None

# ---------------------------------------------------------
# Sidebar Navigation & Workspace Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="font-size: 28px;">👥</span>
        <div>
            <h2>TalentPulse AI</h2>
            <div style="font-size: 12px; color: #64748b; font-weight: 600;">Workforce & Attrition System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page_options = [
        "Dashboard",
        "Add New Employee",
        "Employee Directory",
        "AI Attrition Prediction",
        "Employee Analytics",
        "Model Performance"
    ]

    # Handle external page redirects cleanly
    if st.session_state.nav_page in page_options:
        default_index = page_options.index(st.session_state.nav_page)
    else:
        default_index = 0

    page = st.radio(
        "Navigation",
        page_options,
        index=default_index,
        format_func=lambda x: {
            "Dashboard": "🏠  HR Dashboard",
            "Add New Employee": "➕  Add New Employee",
            "Employee Directory": "📋  Employee Directory",
            "AI Attrition Prediction": "🔮  AI Attrition Prediction",
            "Employee Analytics": "📊  Employee Analytics",
            "Model Performance": "📈  Model Performance"
        }.get(x, x),
        key="sidebar_nav_radio"
    )
    # Sync state
    st.session_state.nav_page = page

    st.divider()

    # Live Database Quick Stats in Sidebar
    db_employees = database.get_all_employees()
    live_count = len(db_employees)
    high_risk_count = len(db_employees[db_employees["predicted_risk"] == "High"]) if live_count > 0 else 0

    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; font-size: 13px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #64748b; font-weight: 600;">Live Registered:</span>
            <span style="font-weight: 700; color: #0f172a;">{live_count} Employees</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #64748b; font-weight: 600;">High Risk Alerts:</span>
            <span style="font-weight: 700; color: #ef4444;">{high_risk_count} Employees</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #64748b; font-weight: 600;">Model Engine:</span>
            <span style="font-weight: 700; color: #10b981;">Logistic Reg (v2)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Database Tools
    with st.expander("🛠️ Database Admin Tools"):
        st.caption("Manage SQLite Database records (`hr_attrition.db`)")
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("🔄 Reset Demo", use_container_width=True, help="Load 6 diverse demo employees"):
                conn = database.get_connection()
                conn.cursor().execute("DELETE FROM employees")
                conn.cursor().execute("DELETE FROM prediction_history")
                conn.commit()
                conn.close()
                database.seed_sample_employees()
                st.toast("Demo data reloaded!", icon="✅")
                st.rerun()
        with col_db2:
            if st.button("🗑️ Clear All", use_container_width=True, help="Wipe all employee records"):
                conn = database.get_connection()
                conn.cursor().execute("DELETE FROM employees")
                conn.cursor().execute("DELETE FROM prediction_history")
                conn.commit()
                conn.close()
                st.toast("Database cleared", icon="⚠️")
                st.rerun()

    st.caption("🏆 Hackathon AI Project 2026\nStreamlit • Scikit-Learn • SQLite")


# =========================================================
# PAGE 1: HR DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Workforce Overview & Risk Dashboard</div>
        <div class="hero-subtitle">Real-time HR analytics, retention tracking, and AI-driven attrition risk distribution.</div>
    </div>
    """, unsafe_allow_html=True)

    # Mode Selector: Live Registered Employees vs Historical IBM Benchmark
    top_c1, top_c2 = st.columns([2, 1])
    with top_c1:
        st.subheader("Workforce Intelligence Metrics")
    with top_c2:
        dataset_mode = st.radio(
            "Data Source View:",
            ["Live Registered Database (SQLite)", "Historical Benchmark (IBM Dataset)"],
            horizontal=True,
            label_visibility="collapsed"
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if dataset_mode == "Live Registered Database (SQLite)":
        df_live = database.get_all_employees()
        total_emp = len(df_live)

        if total_emp == 0:
            st.warning("No employees are registered in the live database yet. Add an employee or click 'Reset Demo' in the sidebar!")
            if st.button("➕ Add First Employee", type="primary"):
                st.session_state.nav_page = "Add New Employee"
                st.rerun()
        else:
            # Live Metrics calculation
            avg_salary = df_live["MonthlyIncome"].mean() if "MonthlyIncome" in df_live.columns and total_emp > 0 else 0
            avg_satisfaction = df_live["JobSatisfaction"].mean() if "JobSatisfaction" in df_live.columns and total_emp > 0 else 0
            
            risk_counts = df_live["predicted_risk"].value_counts().to_dict()
            low_risk = risk_counts.get("Low", 0)
            med_risk = risk_counts.get("Medium", 0)
            high_risk = risk_counts.get("High", 0)
            unassessed = risk_counts.get("Unassessed", 0)

            # Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Registered Staff", f"{total_emp:,}", help="Total active employees in live SQLite database")
            m2.metric("Avg Monthly Salary", f"${avg_salary:,.0f}", help="Average gross monthly income")
            m3.metric("Job Satisfaction", f"{avg_satisfaction:.2f} / 4.0", help="Average job satisfaction rating")
            m4.metric("High Attrition Risk", f"{high_risk}", delta=f"{high_risk/total_emp*100:.1f}% of staff", delta_color="inverse")
            m5.metric("Low Risk (Safe)", f"{low_risk}", delta=f"{low_risk/total_emp*100:.1f}% retained", delta_color="normal")

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

            # Charts Row 1: Department Distribution & Risk Category Distribution
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 🏢 Department-Wise Staff Distribution")
                dept_counts = df_live["Department"].value_counts().reset_index()
                dept_counts.columns = ["Department", "Count"]

                fig_dept = px.pie(
                    dept_counts,
                    names="Department",
                    values="Count",
                    color="Department",
                    hole=0.45,
                    color_discrete_map={
                        "Research & Development": "#3b82f6",
                        "Sales": "#8b5cf6",
                        "Human Resources": "#ec4899"
                    }
                )
                fig_dept.update_traces(textposition='inside', textinfo='percent+label')
                fig_dept.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    height=300
                )
                st.plotly_chart(fig_dept, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 🛡️ AI Predicted Attrition Risk Tiers")
                risk_df = pd.DataFrame({
                    "Risk Tier": ["Low Risk (<30%)", "Medium Risk (30-60%)", "High Risk (>=60%)", "Unassessed"],
                    "Count": [low_risk, med_risk, high_risk, unassessed]
                })
                risk_df = risk_df[risk_df["Count"] > 0]

                fig_risk = px.bar(
                    risk_df,
                    x="Risk Tier",
                    y="Count",
                    color="Risk Tier",
                    text="Count",
                    color_discrete_map={
                        "Low Risk (<30%)": "#10b981",
                        "Medium Risk (30-60%)": "#f59e0b",
                        "High Risk (>=60%)": "#ef4444",
                        "Unassessed": "#94a3b8"
                    }
                )
                fig_risk.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Employees",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Charts Row 2: Registration Timeline & Job Role Compensation
            c3, c4 = st.columns(2)

            with c3:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 📅 Employee Registration Trends (SQLite)")
                if "created_at" in df_live.columns and df_live["created_at"].notnull().any():
                    timeline_df = df_live.copy()
                    timeline_df["reg_date"] = pd.to_datetime(timeline_df["created_at"], errors="coerce").dt.date
                    trend_counts = timeline_df.groupby("reg_date").size().reset_index(name="Registrations")
                    trend_counts["Cumulative"] = trend_counts["Registrations"].cumsum()

                    fig_trend = px.line(
                        trend_counts,
                        x="reg_date",
                        y="Cumulative",
                        markers=True,
                        line_shape="spline",
                        color_discrete_sequence=["#2563eb"]
                    )
                    fig_trend.update_layout(
                        xaxis_title="Registration Date",
                        yaxis_title="Cumulative Employees",
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=280
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No registration date information available.")
                st.markdown("</div>", unsafe_allow_html=True)

            with c4:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 💼 Average Salary by Job Role")
                role_sal = df_live.groupby("JobRole")["MonthlyIncome"].mean().reset_index().sort_values("MonthlyIncome", ascending=True)
                fig_role = px.bar(
                    role_sal,
                    y="JobRole",
                    x="MonthlyIncome",
                    orientation="h",
                    color="MonthlyIncome",
                    color_continuous_scale="Blues"
                )
                fig_role.update_layout(
                    yaxis_title="",
                    xaxis_title="Average Monthly Salary ($)",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=280,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_role, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # High Risk Immediate Attention Callout Table
            if high_risk > 0:
                st.markdown("#### 🚨 High Risk Attention List")
                high_risk_df = df_live[df_live["predicted_risk"] == "High"][
                    ["emp_id", "name", "Department", "JobRole", "MonthlyIncome", "JobSatisfaction", "attrition_score"]
                ].copy()
                high_risk_df["attrition_score"] = high_risk_df["attrition_score"].apply(lambda s: f"{s*100:.1f}%" if pd.notnull(s) else "N/A")
                high_risk_df.columns = ["ID", "Name", "Department", "Role", "Monthly Income ($)", "Job Satisfaction (1-4)", "Attrition Probability"]
                st.dataframe(high_risk_df, use_container_width=True, hide_index=True)

    else:
        # Benchmark Historical IBM Dataset Stats
        st.info("🏛️ **Historical Training Dataset Benchmark:** Viewing statistics from the IBM HR Analytics dataset (1,470 records). This serves as the baseline ground truth used to train our Machine Learning models.")
        
        if not historical_df.empty:
            h_total = len(historical_df)
            h_left = (historical_df["Attrition"] == "Yes").sum()
            h_rate = (h_left / h_total) * 100
            h_avg_sal = historical_df["MonthlyIncome"].mean()
            h_avg_sat = historical_df["JobSatisfaction"].mean()

            hb1, hb2, hb3, hb4, hb5 = st.columns(5)
            hb1.metric("Historical Benchmark Records", f"{h_total:,}")
            hb2.metric("Benchmark Resignations", f"{h_left:,}")
            hb3.metric("Baseline Attrition Rate", f"{h_rate:.1f}%")
            hb4.metric("Avg Benchmark Salary", f"${h_avg_sal:,.0f}")
            hb5.metric("Avg Satisfaction", f"{h_avg_sat:.2f} / 4.0")

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 🔬 Benchmark Attrition Breakdown")
                att_counts = historical_df["Attrition"].value_counts().reset_index()
                att_counts.columns = ["Attrition", "Count"]
                fig_h1 = px.pie(
                    att_counts,
                    names="Attrition",
                    values="Count",
                    hole=0.45,
                    color="Attrition",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"}
                )
                fig_h1.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_h1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### ⏰ Overtime vs Attrition in Historical Data")
                ot_att = pd.crosstab(historical_df["OverTime"], historical_df["Attrition"], normalize='index') * 100
                ot_att_reset = ot_att.reset_index().melt(id_vars="OverTime", value_name="Percentage", var_name="Attrition")
                fig_ot = px.bar(
                    ot_att_reset,
                    x="OverTime",
                    y="Percentage",
                    color="Attrition",
                    barmode="group",
                    color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"},
                    text_auto=".1f"
                )
                fig_ot.update_layout(
                    yaxis_title="Percentage (%)",
                    height=300,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_ot, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Historical dataset not found.")


# =========================================================
# PAGE 2: ADD NEW EMPLOYEE
# =========================================================
elif page == "Add New Employee":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Register New Employee</div>
        <div class="hero-subtitle">Input employee profile, role details, and compensation. Data is saved directly to SQLite.</div>
    </div>
    """, unsafe_allow_html=True)

    next_id = database.get_next_emp_id()

    with st.form("add_employee_form", clear_on_submit=False):
        st.markdown("### 👤 Section 1: Basic Profile & Identification")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            emp_id = st.text_input("Employee ID", value=next_id, disabled=True, help="Auto-generated sequential employee ID")
        with c2:
            name = st.text_input("Full Name *", placeholder="e.g., Alexandra Rivera", help="Employee's official legal name")
        with c3:
            gender = st.selectbox("Gender", ml_utils.CATEGORICAL_OPTIONS["Gender"], index=0)

        c4, c5, c6 = st.columns(3)
        with c4:
            age = st.number_input("Age (Years) *", min_value=18, max_value=65, value=32, step=1)
        with c5:
            marital_status = st.selectbox("Marital Status", ml_utils.CATEGORICAL_OPTIONS["MaritalStatus"], index=1)
        with c6:
            distance = st.number_input("Distance From Home (miles/km)", min_value=1, max_value=60, value=8, step=1)

        st.divider()

        st.markdown("### 🏢 Section 2: Department, Role & Compensation")
        r1, r2, r3 = st.columns(3)
        with r1:
            department = st.selectbox("Department *", ml_utils.CATEGORICAL_OPTIONS["Department"], index=1)
        with r2:
            job_role = st.selectbox("Job Role *", ml_utils.CATEGORICAL_OPTIONS["JobRole"], index=6)
        with r3:
            monthly_income = st.number_input("Monthly Income ($) *", min_value=1000.0, max_value=50000.0, value=5500.0, step=100.0)

        r4, r5, r6 = st.columns(3)
        with r4:
            job_level = st.selectbox("Job Level (1 = Entry, 5 = Executive)", [1, 2, 3, 4, 5], index=1)
        with r5:
            stock_option = st.selectbox("Stock Option Level (0 = None, 3 = High)", [0, 1, 2, 3], index=1)
        with r6:
            salary_hike = st.slider("Percent Salary Hike (%)", min_value=10, max_value=30, value=14)

        st.divider()

        st.markdown("### 🌟 Section 3: Workplace Experience & Satisfaction")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            job_satisfaction = st.select_slider("Job Satisfaction (1-4) *", options=[1, 2, 3, 4], value=3, help="1: Low, 2: Medium, 3: High, 4: Very High")
        with s2:
            work_life_balance = st.select_slider("Work-Life Balance (1-4) *", options=[1, 2, 3, 4], value=3, help="1: Bad, 2: Good, 3: Better, 4: Best")
        with s3:
            env_satisfaction = st.select_slider("Environment Satisfaction (1-4)", options=[1, 2, 3, 4], value=3)
        with s4:
            job_involvement = st.select_slider("Job Involvement (1-4)", options=[1, 2, 3, 4], value=3)

        st.divider()

        st.markdown("### ⏳ Section 4: Tenure & Work Arrangement")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            overtime = st.selectbox("Overtime Required? *", ml_utils.CATEGORICAL_OPTIONS["OverTime"], index=0)
        with t2:
            total_working_years = st.number_input("Total Working Years *", min_value=0, max_value=50, value=8, step=1)
        with t3:
            years_at_company = st.number_input("Years at Company *", min_value=0, max_value=40, value=4, step=1)
        with t4:
            years_since_promotion = st.number_input("Years Since Last Promotion *", min_value=0, max_value=20, value=1, step=1)

        t5, t6, t7 = st.columns(3)
        with t5:
            years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=25, value=2, step=1)
        with t6:
            years_with_manager = st.number_input("Years with Current Manager", min_value=0, max_value=25, value=2, step=1)
        with t7:
            business_travel = st.selectbox("Business Travel Frequency", ml_utils.CATEGORICAL_OPTIONS["BusinessTravel"], index=2)

        # Advanced Details Expander for model defaults
        with st.expander("⚙️ Advanced Model Features (Dataset Defaults Provided)", expanded=False):
            st.caption("These features are used by the AI model. They are pre-filled with the IBM dataset medians and can be adjusted if needed.")
            ad1, ad2, ad3 = st.columns(3)
            with ad1:
                daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=2000, value=802, step=10)
                hourly_rate = st.number_input("Hourly Rate ($)", min_value=20, max_value=150, value=66, step=1)
                monthly_rate = st.number_input("Monthly Rate ($)", min_value=1000, max_value=30000, value=14235, step=100)
            with ad2:
                education = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=2, format_func=lambda x: f"{x} - {['Below College', 'College', 'Bachelor', 'Master', 'Doctor'][x-1]}")
                education_field = st.selectbox("Education Field", ml_utils.CATEGORICAL_OPTIONS["EducationField"], index=1)
                num_companies = st.number_input("Number of Companies Worked", min_value=0, max_value=15, value=2, step=1)
            with ad3:
                training_times = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=3, step=1)
                performance_rating = st.selectbox("Performance Rating", [3, 4], index=0, format_func=lambda x: "3 - Excellent" if x == 3 else "4 - Outstanding")
                rel_satisfaction = st.select_slider("Relationship Satisfaction", options=[1, 2, 3, 4], value=3)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("💾 Save Employee Record", type="primary", use_container_width=True)

    if submit_btn:
        # 1. Validation Logic
        errors = []
        if not name.strip():
            errors.append("Employee Full Name is required.")
        if years_at_company > total_working_years:
            errors.append(f"Years at Company ({years_at_company}) cannot exceed Total Working Years ({total_working_years}).")
        if years_in_role > years_at_company:
            errors.append(f"Years in Current Role ({years_in_role}) cannot exceed Years at Company ({years_at_company}).")
        if years_since_promotion > years_at_company:
            errors.append(f"Years Since Last Promotion ({years_since_promotion}) cannot exceed Years at Company ({years_at_company}).")
        if years_with_manager > years_at_company:
            errors.append(f"Years with Manager ({years_with_manager}) cannot exceed Years at Company ({years_at_company}).")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            # Prepare data payload
            employee_record = {
                "emp_id": next_id,
                "name": name.strip(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Age": int(age),
                "BusinessTravel": business_travel,
                "DailyRate": int(daily_rate),
                "Department": department,
                "DistanceFromHome": int(distance),
                "Education": int(education),
                "EducationField": education_field,
                "EnvironmentSatisfaction": int(env_satisfaction),
                "Gender": gender,
                "HourlyRate": int(hourly_rate),
                "JobInvolvement": int(job_involvement),
                "JobLevel": int(job_level),
                "JobRole": job_role,
                "JobSatisfaction": int(job_satisfaction),
                "MaritalStatus": marital_status,
                "MonthlyIncome": float(monthly_income),
                "MonthlyRate": int(monthly_rate),
                "NumCompaniesWorked": int(num_companies),
                "OverTime": overtime,
                "PercentSalaryHike": int(salary_hike),
                "PerformanceRating": int(performance_rating),
                "RelationshipSatisfaction": int(rel_satisfaction),
                "StockOptionLevel": int(stock_option),
                "TotalWorkingYears": int(total_working_years),
                "TrainingTimesLastYear": int(training_times),
                "WorkLifeBalance": int(work_life_balance),
                "YearsAtCompany": int(years_at_company),
                "YearsInCurrentRole": int(years_in_role),
                "YearsSinceLastPromotion": int(years_since_promotion),
                "YearsWithCurrManager": int(years_with_manager)
            }

            # 2. Run instant ML prediction baseline
            try:
                pred_result = ml_utils.predict_attrition(employee_record)
                employee_record["predicted_risk"] = pred_result["risk_category"]
                employee_record["attrition_score"] = pred_result["attrition_score"]
                employee_record["prediction_label"] = pred_result["classification"]
                employee_record["last_prediction_date"] = employee_record["created_at"]
            except Exception as e:
                employee_record["predicted_risk"] = "Unassessed"
                pred_result = None

            # 3. Save to SQLite
            new_row_id = database.add_employee(employee_record)

            if pred_result:
                database.log_prediction_history(
                    next_id,
                    employee_record["name"],
                    pred_result["attrition_score"],
                    pred_result["risk_category"],
                    pred_result["classification"],
                    employee_record["created_at"],
                    pred_result["risk_drivers"]
                )

            st.balloons()
            st.success(f"🎉 Employee **{name.strip()}** (ID: `{next_id}`) saved successfully to the database!")

            # Quick routing buttons
            post_c1, post_c2 = st.columns(2)
            with post_c1:
                if st.button("📋 Open Employee Directory", use_container_width=True):
                    st.session_state.nav_page = "Employee Directory"
                    st.rerun()
            with post_c2:
                if st.button(f"🔮 View AI Attrition Analysis for {next_id}", use_container_width=True):
                    st.session_state.selected_emp_id = next_id
                    st.session_state.nav_page = "AI Attrition Prediction"
                    st.rerun()


# =========================================================
# PAGE 3: EMPLOYEE DIRECTORY
# =========================================================
elif page == "Employee Directory":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Employee Directory & Record Management</div>
        <div class="hero-subtitle">Search, filter, view details, edit records, delete, and export live employees to CSV.</div>
    </div>
    """, unsafe_allow_html=True)

    df_employees = database.get_all_employees()

    if df_employees.empty:
        st.info("No employees registered yet. Please add a new employee or reload demo records from the sidebar.")
    else:
        # Search & Filter Toolbar
        f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1.5])
        with f1:
            search_query = st.text_input("🔍 Search by Name or ID", placeholder="Search employee...", label_visibility="collapsed")
        with f2:
            dept_filter = st.selectbox(
                "Filter Department",
                ["All Departments"] + sorted(df_employees["Department"].dropna().unique().tolist()),
                label_visibility="collapsed"
            )
        with f3:
            risk_filter = st.selectbox(
                "Filter Risk Tier",
                ["All Risk Tiers", "Low", "Medium", "High", "Unassessed"],
                label_visibility="collapsed"
            )
        with f4:
            # Export CSV Button
            csv_buffer = io.StringIO()
            df_employees.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export CSV",
                data=csv_buffer.getvalue(),
                file_name=f"employee_directory_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Apply Filters
        filtered_df = df_employees.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["name"].str.contains(search_query, case=False, na=False) |
                filtered_df["emp_id"].str.contains(search_query, case=False, na=False)
            ]
        if dept_filter != "All Departments":
            filtered_df = filtered_df[filtered_df["Department"] == dept_filter]
        if risk_filter != "All Risk Tiers":
            filtered_df = filtered_df[filtered_df["predicted_risk"] == risk_filter]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(df_employees)}** employees")

        # Table Display
        display_cols = ["emp_id", "name", "Department", "JobRole", "MonthlyIncome", "JobSatisfaction", "predicted_risk", "attrition_score"]
        table_view = filtered_df[display_cols].copy()
        table_view["MonthlyIncome"] = table_view["MonthlyIncome"].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "$0")
        table_view["JobSatisfaction"] = table_view["JobSatisfaction"].apply(lambda x: f"{x}/4 ⭐" if pd.notnull(x) else "-")
        table_view["attrition_score"] = table_view["attrition_score"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
        table_view.columns = ["Employee ID", "Full Name", "Department", "Job Role", "Monthly Income", "Satisfaction", "Risk Tier", "Attrition Probability"]

        st.dataframe(
            table_view,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Employee Action & Profile Panel
        st.markdown("### 🛠️ Employee Record Actions")
        
        emp_options = filtered_df["emp_id"].tolist()
        if not emp_options:
            st.info("No matching employees found for selected filters.")
        else:
            selected_id = st.selectbox(
                "Select Employee to Manage:",
                emp_options,
                format_func=lambda x: f"[{x}] {df_employees.loc[df_employees['emp_id']==x, 'name'].values[0]} ({df_employees.loc[df_employees['emp_id']==x, 'JobRole'].values[0]})"
            )

            current_emp = database.get_employee(selected_id)

            action_tabs = st.tabs(["👁️ View Full Profile", "✏️ Edit Details", "🔮 Run AI Prediction", "🗑️ Delete Record"])

            # TAB 1: VIEW PROFILE
            with action_tabs[0]:
                if current_emp:
                    risk_badge_class = f"badge-{current_emp.get('predicted_risk', 'unassessed').lower()}"
                    risk_text = current_emp.get("predicted_risk", "Unassessed")
                    score_val = current_emp.get("attrition_score")
                    score_str = f"{score_val*100:.1f}%" if score_val is not None else "Not Assessed"

                    st.markdown(f"""
                    <div class="card-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                            <div>
                                <h3 style="margin: 0; color: #1e3a8a;">{current_emp['name']}</h3>
                                <div style="color: #64748b; font-size: 14px;">Employee ID: <b>{current_emp['emp_id']}</b> • Registered: {current_emp.get('created_at', 'N/A')}</div>
                            </div>
                            <div>
                                <span class="{risk_badge_class}">Risk Level: {risk_text} ({score_str})</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.markdown("**Role & Compensation**")
                        st.write(f"• **Department:** {current_emp.get('Department')}")
                        st.write(f"• **Job Role:** {current_emp.get('JobRole')}")
                        st.write(f"• **Monthly Salary:** ${current_emp.get('MonthlyIncome', 0):,.2f}")
                        st.write(f"• **Job Level:** {current_emp.get('JobLevel')}/5")
                        st.write(f"• **Stock Option:** {current_emp.get('StockOptionLevel')}/3")
                    with p2:
                        st.markdown("**Tenure & Experience**")
                        st.write(f"• **Years at Company:** {current_emp.get('YearsAtCompany')} yrs")
                        st.write(f"• **Total Working Years:** {current_emp.get('TotalWorkingYears')} yrs")
                        st.write(f"• **Years in Current Role:** {current_emp.get('YearsInCurrentRole')} yrs")
                        st.write(f"• **Years Since Promotion:** {current_emp.get('YearsSinceLastPromotion')} yrs")
                        st.write(f"• **Years with Manager:** {current_emp.get('YearsWithCurrManager')} yrs")
                    with p3:
                        st.markdown("**Workplace Conditions**")
                        st.write(f"• **Job Satisfaction:** {current_emp.get('JobSatisfaction')}/4")
                        st.write(f"• **Work-Life Balance:** {current_emp.get('WorkLifeBalance')}/4")
                        st.write(f"• **Overtime Required:** {current_emp.get('OverTime')}")
                        st.write(f"• **Business Travel:** {current_emp.get('BusinessTravel')}")
                        st.write(f"• **Commute Distance:** {current_emp.get('DistanceFromHome')} miles")

            # TAB 2: EDIT EMPLOYEE
            with action_tabs[1]:
                if current_emp:
                    with st.form(f"edit_form_{selected_id}"):
                        st.write(f"Editing profile for **{current_emp['name']}** (`{selected_id}`)")
                        e1, e2 = st.columns(2)
                        with e1:
                            new_name = st.text_input("Full Name", value=current_emp["name"])
                            new_dept = st.selectbox("Department", ml_utils.CATEGORICAL_OPTIONS["Department"], index=ml_utils.CATEGORICAL_OPTIONS["Department"].index(current_emp["Department"]) if current_emp["Department"] in ml_utils.CATEGORICAL_OPTIONS["Department"] else 0)
                            new_role = st.selectbox("Job Role", ml_utils.CATEGORICAL_OPTIONS["JobRole"], index=ml_utils.CATEGORICAL_OPTIONS["JobRole"].index(current_emp["JobRole"]) if current_emp["JobRole"] in ml_utils.CATEGORICAL_OPTIONS["JobRole"] else 0)
                            new_salary = st.number_input("Monthly Income ($)", value=float(current_emp.get("MonthlyIncome", 5000.0)), step=100.0)
                        with e2:
                            new_ot = st.selectbox("Overtime", ml_utils.CATEGORICAL_OPTIONS["OverTime"], index=0 if current_emp.get("OverTime")=="No" else 1)
                            new_js = st.select_slider("Job Satisfaction (1-4)", options=[1, 2, 3, 4], value=int(current_emp.get("JobSatisfaction", 3)))
                            new_wlb = st.select_slider("Work-Life Balance (1-4)", options=[1, 2, 3, 4], value=int(current_emp.get("WorkLifeBalance", 3)))
                            new_yac = st.number_input("Years at Company", min_value=0, max_value=40, value=int(current_emp.get("YearsAtCompany", 1)))

                        edit_submit = st.form_submit_btn("💾 Update Employee", type="primary")

                    if edit_submit:
                        updated_payload = {
                            "name": new_name.strip(),
                            "Department": new_dept,
                            "JobRole": new_role,
                            "MonthlyIncome": float(new_salary),
                            "OverTime": new_ot,
                            "JobSatisfaction": int(new_js),
                            "WorkLifeBalance": int(new_wlb),
                            "YearsAtCompany": int(new_yac)
                        }
                        # Re-calculate prediction
                        full_emp = {**current_emp, **updated_payload}
                        pred_re = ml_utils.predict_attrition(full_emp)
                        updated_payload["predicted_risk"] = pred_re["risk_category"]
                        updated_payload["attrition_score"] = pred_re["attrition_score"]
                        updated_payload["prediction_label"] = pred_re["classification"]
                        updated_payload["last_prediction_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        database.update_employee(selected_id, updated_payload)
                        st.success(f"Record for {new_name} updated successfully!")
                        st.rerun()

            # TAB 3: RUN PREDICTION SHORTCUT
            with action_tabs[2]:
                st.write(f"Jump to the AI Prediction Engine for **{current_emp['name']}** (`{selected_id}`)")
                if st.button("🚀 Open in AI Attrition Prediction", type="primary"):
                    st.session_state.selected_emp_id = selected_id
                    st.session_state.nav_page = "AI Attrition Prediction"
                    st.rerun()

            # TAB 4: DELETE EMPLOYEE
            with action_tabs[3]:
                st.error(f"⚠️ **Danger Zone:** Permanently remove Employee **{current_emp['name']}** (`{selected_id}`)")
                st.write("Deleting this employee will remove all associated profile data and prediction logs.")
                confirm_del = st.checkbox(f"Yes, I confirm deleting {selected_id} ({current_emp['name']})")
                if st.button("🗑️ Delete Employee Permanently", type="secondary", disabled=not confirm_del):
                    database.delete_employee(selected_id)
                    st.toast(f"Employee {selected_id} deleted.", icon="🗑️")
                    st.rerun()


# =========================================================
# PAGE 4: AI ATTRITION PREDICTION
# =========================================================
elif page == "AI Attrition Prediction":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">AI Attrition Risk Predictor</div>
        <div class="hero-subtitle">Evaluate retention probability using attrition_model_v2.pkl with transparent risk factors and HR action plans.</div>
    </div>
    """, unsafe_allow_html=True)

    df_employees = database.get_all_employees()

    mode = st.radio(
        "Prediction Mode:",
        ["Select Saved Employee from Database", "Interactive Custom Scenario Simulator"],
        horizontal=True
    )

    emp_input_data = {}
    display_name = "Custom Simulation Candidate"
    current_emp_id = None

    if mode == "Select Saved Employee from Database":
        if df_employees.empty:
            st.warning("No employees registered yet. Please add an employee or use the simulator.")
            current_emp_dict = None
        else:
            default_idx = 0
            if st.session_state.selected_emp_id in df_employees["emp_id"].values:
                default_idx = df_employees["emp_id"].tolist().index(st.session_state.selected_emp_id)

            chosen_id = st.selectbox(
                "Choose Employee to Evaluate:",
                df_employees["emp_id"].tolist(),
                index=default_idx,
                format_func=lambda x: f"[{x}] {df_employees.loc[df_employees['emp_id']==x, 'name'].values[0]} — {df_employees.loc[df_employees['emp_id']==x, 'JobRole'].values[0]} ({df_employees.loc[df_employees['emp_id']==x, 'Department'].values[0]})"
            )
            current_emp_id = chosen_id
            current_emp_dict = database.get_employee(chosen_id)
            if current_emp_dict:
                display_name = current_emp_dict["name"]
                emp_input_data = current_emp_dict

                # Display summary card of loaded employee
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 16px; margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 18px; font-weight: 700; color: #0f172a;">{display_name}</span>
                            <span style="color: #64748b; font-size: 14px;"> (ID: <b>{chosen_id}</b>)</span>
                            <div style="font-size: 13px; color: #475569; margin-top: 4px;">
                                {current_emp_dict.get('Department')} • {current_emp_dict.get('JobRole')} • Salary: ${current_emp_dict.get('MonthlyIncome', 0):,.0f} • Overtime: {current_emp_dict.get('OverTime')}
                            </div>
                        </div>
                        <div>
                            <span style="font-size: 13px; color: #64748b;">Previous Assessment:</span>
                            <b>{current_emp_dict.get('predicted_risk', 'Unassessed')}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Interactive Simulator Mode
        st.info("💡 Adjust any parameter below to perform real-time 'What-If' sensitivity analysis on attrition propensity.")
        display_name = st.text_input("Simulation Candidate Name", value="Jordan Lee (Scenario Test)")
        
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            s_age = st.number_input("Age", min_value=18, max_value=65, value=29)
            s_dept = st.selectbox("Department", ml_utils.CATEGORICAL_OPTIONS["Department"], index=1)
            s_role = st.selectbox("Job Role", ml_utils.CATEGORICAL_OPTIONS["JobRole"], index=2)
            s_salary = st.number_input("Monthly Income ($)", min_value=1000.0, max_value=50000.0, value=3200.0)
        with sim_c2:
            s_ot = st.selectbox("Overtime", ml_utils.CATEGORICAL_OPTIONS["OverTime"], index=1)
            s_js = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=1)
            s_wlb = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=2)
            s_travel = st.selectbox("Business Travel", ml_utils.CATEGORICAL_OPTIONS["BusinessTravel"], index=1)
        with sim_c3:
            s_yac = st.number_input("Years at Company", min_value=0, max_value=40, value=2)
            s_twy = st.number_input("Total Working Years", min_value=0, max_value=45, value=4)
            s_yslp = st.number_input("Years Since Promotion", min_value=0, max_value=15, value=2)
            s_dist = st.number_input("Distance From Home", min_value=1, max_value=50, value=22)

        emp_input_data = {
            **ml_utils.DATASET_DEFAULTS,
            "Age": int(s_age),
            "Department": s_dept,
            "JobRole": s_role,
            "MonthlyIncome": float(s_salary),
            "OverTime": s_ot,
            "JobSatisfaction": int(s_js),
            "WorkLifeBalance": int(s_wlb),
            "BusinessTravel": s_travel,
            "YearsAtCompany": int(s_yac),
            "TotalWorkingYears": int(s_twy),
            "YearsSinceLastPromotion": int(s_yslp),
            "DistanceFromHome": int(s_dist)
        }

    # Prediction Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_click = st.button("⚡ Run AI Attrition Prediction", type="primary", use_container_width=True)

    if predict_click:
        if not emp_input_data:
            st.error("No valid employee data available to predict.")
        else:
            with st.spinner("Analyzing workforce features against trained Logistic Regression pipeline..."):
                try:
                    result = ml_utils.predict_attrition(emp_input_data)
                    score = result["attrition_score"]
                    risk_cat = result["risk_category"]
                    classification = result["classification"]
                    pred_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # If this is a saved employee, persist to SQLite!
                    if current_emp_id:
                        database.update_prediction(
                            current_emp_id,
                            score,
                            risk_cat,
                            classification,
                            pred_date
                        )
                        database.log_prediction_history(
                            current_emp_id,
                            display_name,
                            score,
                            risk_cat,
                            classification,
                            pred_date,
                            result["risk_drivers"]
                        )

                    # Display Rich Results Card
                    st.divider()
                    st.markdown("### 🎯 Prediction Analysis Results")

                    res_col1, res_col2 = st.columns([1.2, 1])

                    with res_col1:
                        st.markdown(f"""
                        <div class="card-box" style="border-left: 6px solid {result['risk_color']};">
                            <div style="font-size: 13px; color: #64748b; font-weight: 700; text-transform: uppercase;">Employee Evaluation</div>
                            <div style="font-size: 24px; font-weight: 800; color: #0f172a; margin: 4px 0;">{display_name}</div>
                            <div style="display: flex; gap: 10px; align-items: center; margin: 12px 0;">
                                <span class="badge-{risk_cat.lower()}" style="font-size: 14px; padding: 6px 14px;">
                                    Illustrative Risk: {risk_cat.upper()} RISK
                                </span>
                                <span style="font-weight: 700; font-size: 14px; color: {'#991b1b' if result['binary_prediction']==1 else '#065f46'};">
                                    • {classification}
                                </span>
                            </div>
                            <div style="margin-top: 15px;">
                                <div style="display: flex; justify-content: space-between; font-size: 14px; font-weight: 700; margin-bottom: 4px;">
                                    <span>Attrition Propensity Score:</span>
                                    <span>{score * 100:.1f}%</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Streamlit native progress bar
                        st.progress(float(np.clip(score, 0.0, 1.0)))

                        st.caption(f"📅 Assessment Timestamp: {pred_date} • Model: LogisticRegression (Class-Balanced Pipeline)")

                    with res_col2:
                        # Plotly Gauge Chart for visual WOW
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Attrition Risk Gauge (%)", 'font': {'size': 16, 'color': '#1e293b'}},
                            number={'suffix': "%", 'font': {'size': 28, 'color': '#0f172a'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                                'bar': {'color': result['risk_color'], 'thickness': 0.3},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "#e2e8f0",
                                'steps': [
                                    {'range': [0, 30], 'color': '#ecfdf5'},
                                    {'range': [30, 60], 'color': '#fffbeb'},
                                    {'range': [60, 100], 'color': '#fef2f2'}
                                ],
                                'threshold': {
                                    'line': {'color': "#0f172a", 'width': 3},
                                    'thickness': 0.8,
                                    'value': score * 100
                                }
                            }
                        ))
                        fig_gauge.update_layout(
                            height=250,
                            margin=dict(t=30, b=10, l=30, r=30)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    # Explainability: Risk Factors & Recommendations
                    st.markdown("<br>", unsafe_allow_html=True)
                    exp_col1, exp_col2 = st.columns(2)

                    with exp_col1:
                        st.markdown("<div class='card-box' style='height: 100%;'>", unsafe_allow_html=True)
                        st.markdown("#### ⚠️ Key Risk Contributors (Driving Attrition)")
                        if result["risk_drivers"]:
                            for driver in result["risk_drivers"]:
                                st.markdown(f"- 🔴 **{driver}**")
                        else:
                            st.markdown("✅ *No major negative risk signals detected for this profile.*")

                        st.markdown("<br>#### 🛡️ Protective Retention Factors", unsafe_allow_html=True)
                        if result["protective_factors"]:
                            for prot in result["protective_factors"]:
                                st.markdown(f"- 🟢 **{prot}**")
                        else:
                            st.markdown("⚠️ *Few protective factors detected; role may feel demanding or unrewarded.*")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with exp_col2:
                        st.markdown("<div class='card-box' style='height: 100%;'>", unsafe_allow_html=True)
                        st.markdown("#### 💡 Actionable HR Retention Plan")
                        st.write("Suggested strategic interventions for HR business partners and line managers:")
                        for idx, rec in enumerate(result["recommendations"], 1):
                            st.markdown(f"**{idx}.** {rec}")

                        st.markdown("---")
                        st.markdown("""
                        <div style="background: #f8fafc; border-left: 4px solid #f59e0b; padding: 12px; font-size: 13px; color: #475569;">
                            <b>⚠️ Model Limitations & Statistical Disclaimer:</b><br>
                            This Machine Learning model was trained with class balancing (<code>class_weight='balanced'</code>) on historical IBM HR data. 
                            Output probabilities reflect relative attrition propensity rather than mathematically calibrated real-world resignation probabilities. 
                            Treat this score as an early-warning signal for supportive HR conversations, not an automated employment judgment.
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")

    # Section 3: Recent Prediction Audit Log
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📜 Prediction History Audit Log")
    hist_df = database.get_prediction_history(limit=15)
    if not hist_df.empty:
        hist_display = hist_df[["emp_id", "name", "prediction_label", "risk_level", "attrition_score", "prediction_date"]].copy()
        hist_display["attrition_score"] = hist_display["attrition_score"].apply(lambda x: f"{x*100:.1f}%")
        hist_display.columns = ["Employee ID", "Full Name", "Model Classification", "Risk Tier", "Attrition Score", "Timestamp"]
        st.dataframe(hist_display, use_container_width=True, hide_index=True)
    else:
        st.caption("No prediction history recorded yet.")


# =========================================================
# PAGE 5: EMPLOYEE ANALYTICS
# =========================================================
elif page == "Employee Analytics":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Workforce Analytics & Deep Insights</div>
        <div class="hero-subtitle">Interactive exploration of salary distributions, satisfaction scores, overtime effects, and attrition patterns.</div>
    </div>
    """, unsafe_allow_html=True)

    analytics_tab1, analytics_tab2 = st.tabs([
        "🏢 Live Registered Workforce (SQLite)",
        "🏛️ Historical Benchmark Patterns (IBM Dataset)"
    ])

    with analytics_tab1:
        df_live = database.get_all_employees()
        if df_live.empty:
            st.info("No registered employees found in SQLite database. Add employees to view live analytics.")
        else:
            la_c1, la_c2 = st.columns(2)
            with la_c1:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 🏢 Department vs Job Satisfaction")
                fig_sat = px.box(
                    df_live,
                    x="Department",
                    y="JobSatisfaction",
                    color="Department",
                    points="all",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_sat.update_layout(yaxis=dict(tickmode='linear', tick0=1, dtick=1, range=[0.5, 4.5]), height=320)
                st.plotly_chart(fig_sat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with la_c2:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 💰 Salary Distribution by Department")
                fig_sal_live = px.violin(
                    df_live,
                    x="Department",
                    y="MonthlyIncome",
                    color="Department",
                    box=True,
                    points="all",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_sal_live.update_layout(height=320, yaxis_title="Monthly Income ($)")
                st.plotly_chart(fig_sal_live, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            la_c3, la_c4 = st.columns(2)
            with la_c3:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### ⏰ Overtime Distribution Across Roles")
                ot_live = pd.crosstab(df_live["JobRole"], df_live["OverTime"]).reset_index()
                fig_ot_live = px.bar(
                    ot_live,
                    y="JobRole",
                    x=["No", "Yes"],
                    orientation="h",
                    barmode="stack",
                    color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"}
                )
                fig_ot_live.update_layout(height=320, xaxis_title="Employee Count", yaxis_title="")
                st.plotly_chart(fig_ot_live, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with la_c4:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 🛡️ Live Risk Tiers vs Average Income")
                risk_sal = df_live.groupby("predicted_risk")["MonthlyIncome"].agg(["mean", "count"]).reset_index()
                fig_risk_sal = px.bar(
                    risk_sal,
                    x="predicted_risk",
                    y="mean",
                    color="predicted_risk",
                    text_auto="$,.0f",
                    color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444", "Unassessed": "#94a3b8"}
                )
                fig_risk_sal.update_layout(height=320, yaxis_title="Average Salary ($)", xaxis_title="Risk Tier")
                st.plotly_chart(fig_risk_sal, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with analytics_tab2:
        if historical_df.empty:
            st.warning("Historical dataset not loaded.")
        else:
            st.caption("Analyzing historical relationships from 1,470 IBM HR benchmark records.")

            hb_c1, hb_c2 = st.columns(2)
            with hb_c1:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 🚪 Attrition Rate by Job Role")
                role_att = historical_df.groupby("JobRole")["Attrition"].apply(lambda s: (s=="Yes").mean()*100).reset_index(name="AttritionRate").sort_values("AttritionRate", ascending=True)
                fig_ha1 = px.bar(
                    role_att,
                    y="JobRole",
                    x="AttritionRate",
                    orientation="h",
                    color="AttritionRate",
                    color_continuous_scale="Reds",
                    text_auto=".1f"
                )
                fig_ha1.update_layout(height=340, xaxis_title="Historical Attrition Rate (%)", yaxis_title="")
                st.plotly_chart(fig_ha1, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with hb_c2:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### ⏳ Years At Company vs Attrition")
                fig_ha2 = px.histogram(
                    historical_df,
                    x="YearsAtCompany",
                    color="Attrition",
                    barmode="overlay",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    opacity=0.75
                )
                fig_ha2.update_layout(height=340, xaxis_title="Years at Company", yaxis_title="Number of Employees")
                st.plotly_chart(fig_ha2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            hb_c3, hb_c4 = st.columns(2)
            with hb_c3:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 🚗 Commute Distance vs Attrition")
                fig_ha3 = px.box(
                    historical_df,
                    x="Attrition",
                    y="DistanceFromHome",
                    color="Attrition",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    points="outliers"
                )
                fig_ha3.update_layout(height=320, yaxis_title="Distance From Home (miles)")
                st.plotly_chart(fig_ha3, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with hb_c4:
                st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                st.markdown("#### 📈 Monthly Income vs Age (Colored by Attrition)")
                fig_ha4 = px.scatter(
                    historical_df,
                    x="Age",
                    y="MonthlyIncome",
                    color="Attrition",
                    color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"},
                    opacity=0.6,
                    trendline="ols"
                )
                fig_ha4.update_layout(height=320, yaxis_title="Monthly Income ($)")
                st.plotly_chart(fig_ha4, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 6: MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Machine Learning Model Performance & Evaluation</div>
        <div class="hero-subtitle">Comprehensive model benchmark, confusion matrix analysis, and explainability feature weights.</div>
    </div>
    """, unsafe_allow_html=True)

    # Hackathon Evaluation Results Table
    eval_data = {
        "Model": ["Logistic Regression (Selected)", "Random Forest Classifier"],
        "Accuracy": [0.7517, 0.8401],
        "Precision": [0.3488, 0.5000],
        "Recall": [0.6383, 0.3404],
        "F1 Score": [0.4511, 0.4051]
    }
    eval_df = pd.DataFrame(eval_data)

    st.markdown("### 📊 Benchmark Comparison on 20% Stratified Test Split")
    
    # Styled metrics table
    formatted_eval = eval_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        formatted_eval[col] = formatted_eval[col].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(formatted_eval, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Interactive Bar Chart of Performance Metrics
    metrics_melted = eval_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_metrics = px.bar(
        metrics_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".2%",
        color_discrete_map={
            "Logistic Regression (Selected)": "#2563eb",
            "Random Forest Classifier": "#94a3b8"
        }
    )
    fig_metrics.update_layout(
        yaxis_title="Score",
        yaxis_range=[0, 1.0],
        height=340,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown("---")

    # Critical Explanation for Hackathon Judges
    st.markdown("### 🏆 Why Logistic Regression Was Selected Over Random Forest")
    
    st.markdown("""
    <div class="card-box" style="border-left: 6px solid #2563eb;">
        <h4 style="color: #1e3a8a; margin-top: 0;">The Imbalanced Data Science Trade-off</h4>
        <p style="color: #334155; line-height: 1.6;">
            In HR attrition prediction, the dataset is inherently imbalanced (~16% attrition, ~84% retention). 
            A naive or uncalibrated classifier can achieve high <b>Accuracy (84%)</b> simply by guessing "Stay" for every employee.
        </p>
        <ul style="color: #334155; line-height: 1.7;">
            <li>
                <b>Random Forest:</b> Achieved 84.0% raw accuracy, but had an unacceptably low <b>Recall of 34.0%</b>. 
                This means it <i>missed 66% of all employees who actually resigned</i>, which defeats the purpose of an early warning system.
            </li>
            <li>
                <b>Logistic Regression with <code>class_weight='balanced'</code>:</b> 
                Achieved a <b>Recall of 63.8%</b> (detecting nearly <b>double</b> the actual resignation cases) 
                and achieved the highest attrition-class <b>F1 Score (0.451 vs 0.405)</b>.
            </li>
        </ul>
        <div style="background: #eff6ff; border-radius: 8px; padding: 12px; margin-top: 10px; font-weight: 600; color: #1e40af;">
            💡 Decision Rationale: For proactive retention, a false alarm (checking in with a content employee) is far less costly than a false negative (losing a critical team member without prior warning). Therefore, Logistic Regression was selected as the optimal production pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confusion Matrix Visualization
    st.markdown("### 🧩 Confusion Matrix Analysis (Test Set: N = 294)")
    cm_c1, cm_c2 = st.columns(2)

    with cm_c1:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### Logistic Regression (Selected)")
        # Actual values from evaluation: TP=30, FN=17, FP=56, TN=191
        cm_lr = np.array([[191, 56], [17, 30]])
        fig_cm_lr = px.imshow(
            cm_lr,
            text_auto=True,
            x=["Predicted Stay (0)", "Predicted Leave (1)"],
            y=["Actual Stay (0)", "Actual Leave (1)"],
            color_continuous_scale="Blues",
            labels=dict(color="Count")
        )
        fig_cm_lr.update_layout(height=280, coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_cm_lr, use_container_width=True)
        st.caption("✅ Caught 30 out of 47 resignations (63.8% Recall).")
        st.markdown("</div>", unsafe_allow_html=True)

    with cm_c2:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### Random Forest")
        # Actual values from evaluation: TP=16, FN=31, FP=16, TN=231
        cm_rf = np.array([[231, 16], [31, 16]])
        fig_cm_rf = px.imshow(
            cm_rf,
            text_auto=True,
            x=["Predicted Stay (0)", "Predicted Leave (1)"],
            y=["Actual Stay (0)", "Actual Leave (1)"],
            color_continuous_scale="Greys",
            labels=dict(color="Count")
        )
        fig_cm_rf.update_layout(height=280, coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_cm_rf, use_container_width=True)
        st.caption("❌ Missed 31 out of 47 resignations (only 34.0% Recall).")
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance / Model Coefficients
    st.markdown("### 🔬 Model Coefficients & Feature Attribution")
    st.write("Inspect how individual employee characteristics mathematically shift the Logistic Regression log-odds of attrition:")

    feat_c1, feat_c2 = st.columns(2)

    with feat_c1:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 🔴 Top Risk Escalators (Positive Coefficients)")
        pos_features = pd.DataFrame({
            "Feature": [
                "Role: Lab Technician",
                "Role: Sales Representative",
                "Travel: Frequently",
                "OverTime: Yes",
                "Field: Human Resources",
                "Job Level",
                "Years Since Last Promotion",
                "Role: Human Resources",
                "Num Companies Worked",
                "Marital: Single"
            ],
            "Coefficient": [1.194, 1.121, 0.788, 0.762, 0.721, 0.607, 0.503, 0.464, 0.460, 0.457]
        })
        fig_pos = px.bar(
            pos_features,
            y="Feature",
            x="Coefficient",
            orientation="h",
            color="Coefficient",
            color_continuous_scale="Reds",
            text_auto=".3f"
        )
        fig_pos.update_layout(height=340, yaxis_autorange="reversed", coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pos, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with feat_c2:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.markdown("#### 🟢 Top Retention Anchors (Negative Coefficients)")
        neg_features = pd.DataFrame({
            "Feature": [
                "Role: Research Director",
                "Travel: Non-Travel",
                "OverTime: No",
                "Field: Other",
                "Role: Healthcare Rep",
                "Total Working Years",
                "Marital: Divorced",
                "Years With Curr Manager",
                "Environment Satisfaction",
                "Job Satisfaction"
            ],
            "Coefficient": [-1.466, -0.994, -0.893, -0.798, -0.647, -0.620, -0.482, -0.455, -0.426, -0.389]
        })
        fig_neg = px.bar(
            neg_features,
            y="Feature",
            x="Coefficient",
            orientation="h",
            color="Coefficient",
            color_continuous_scale="Greens_r",
            text_auto=".3f"
        )
        fig_neg.update_layout(height=340, yaxis_autorange="reversed", coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_neg, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Pipeline Architecture
    st.markdown("### ⚙️ Machine Learning Pipeline Architecture")
    st.markdown("""
    ```
    Raw Features (30) 
      ├── Numerical Features (23) ──> StandardScaler() ─────────────┐
      └── Categorical Features (7) ──> OneHotEncoder(handle_unknown) ┴──> LogisticRegression(class_weight='balanced') ──> Prediction & Probabilities
    ```
    """)
    st.caption("Training features exclude: Attrition (Target), EmployeeNumber, EmployeeCount, Over18, and StandardHours (Zero-variance constants / ID columns).")
