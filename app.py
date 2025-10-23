import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ---------------- Page Settings ----------------
st.set_page_config(page_title="Hospital Clinical Departments Performance Dynamic Prediction System", layout="centered")
st.title("🏥 Hospital Clinical Departments Performance Dynamic Prediction System")

# ---------------- System Introduction ----------------
st.markdown("""
The system was built using **explainable machine learning techniques** on historical data across different time windows and output predictions based on performance data from corresponding periods. It helped to monitor clinical department performance trends and supported data-driven management decisions.
""")

# ---------------- Feature List (English Names) ----------------
features = [
    "Proportion of Medical Service Revenue in Total Medical Revenue",
    "Proportion of Consumables in Total Medical Revenue",
    "Proportion of Outpatient Revenue in Total Medical Revenue",
    "Proportion of Outpatient Revenue from Medical Insurance Fund",
    "Proportion of Inpatient Revenue in Total Medical Revenue",
    "Proportion of Inpatient Revenue from Medical Insurance Fund",
    "Average Outpatient Drug Expense per Visit",
    "Average Outpatient Expense per Visit",
    "Average Inpatient Expense per Admission",
    "Average Inpatient Drug Expense per Admission"
]

# ---------------- Department to Category Mapping ----------------
department_mapping = {
    "Department of Pediatrics": "Departments of Obstetrics, Gynecology, and Pediatrics",
    "Department of Otorhinolaryngology-Head and Neck Surgery": "Surgical Departments",
    "Department of Rheumatology and Immunology": "Medical Departments",
    "Department of Obstetrics and Gynecology": "Departments of Obstetrics, Gynecology, and Pediatrics",
    "Department of Hepatobiliary Surgery": "Surgical Departments",
    "Department of Hyperbaric Oxygen and Rehabilitation": "Other Clinical Departments",
    "Department of Orthopedics": "Surgical Departments",
    "Department of Respiratory Medicine": "Medical Departments",
    "Department of Rehabilitation Medicine": "Other Clinical Departments",
    "Department of Stomatology / Department of Dentistry": "Other Clinical Departments",
    "Department of Urology": "Surgical Departments",
    "Department of Endocrinology": "Medical Departments",
    "Department of Dermatology": "Other Clinical Departments",
    "Department of General Surgery": "Surgical Departments",
    "Department of General Practice": "Other Clinical Departments",
    "Department of Burn and Plastic Surgery": "Surgical Departments",
    "Department of Neurology": "Medical Departments",
    "Department of Neurosurgery": "Surgical Departments",
    "Department of Nephrology": "Medical Departments",
    "Department of Gastroenterology": "Medical Departments",
    "Department of Cardiothoracic Surgery": "Surgical Departments",
    "Department of Cardiovascular Medicine": "Medical Departments",
    "Department of Cardiac Surgery": "Surgical Departments",
    "Department of Hematology": "Medical Departments",
    "Department of Ophthalmology": "Other Clinical Departments",
    "Department of Traditional Chinese Medicine": "Other Clinical Departments",
    "Department of Oncology": "Other Clinical Departments"
}

# ---------------- Month Window -> model & shap background file mapping ----------------
window_to_model = {
    "Jan-Mar": "best_model_1_3_LinearRegression.joblib",
    "Jan-Jun": "best_model_1_6_LinearRegression.joblib",
    "Jan-Sep": "best_model_1_9_LinearRegression.joblib",
}
window_to_shap_bg = {
    "Jan-Mar": "shap_background_1_3.csv",
    "Jan-Jun": "shap_background_1_6.csv",
    "Jan-Sep": "shap_background_1_9.csv",
}

# ---------------- Load Quartile Thresholds ----------------
quartile_file = "performance_quartile_ranges_2023.pkl"
if not os.path.exists(quartile_file):
    st.error(f"❌ Quartile threshold file not found: {quartile_file}")
    st.stop()
quartile_ranges = joblib.load(quartile_file)

def map_to_grade(score, quartiles):
    if quartiles["Q4"][0] <= score <= quartiles["Q4"][1]:
        return "A"
    elif quartiles["Q3"][0] <= score < quartiles["Q3"][1]:
        return "B"
    elif quartiles["Q2"][0] <= score < quartiles["Q2"][1]:
        return "C"
    else:
        return "D"

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
department_choice = st.sidebar.selectbox(
    "Select Department",
    list(department_mapping.keys()),
    index=list(department_mapping.keys()).index("Department of Oncology")  
)
department_category = department_mapping[department_choice]
st.sidebar.write(f"🏷️ Department Category: **{department_category}**")

window_choice = st.sidebar.selectbox("Select Month Window", list(window_to_model.keys()))
model_path = window_to_model[window_choice]
shap_bg_path = window_to_shap_bg.get(window_choice, None)

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)

# ---------------- Main Panel: Feature Input ----------------
st.subheader(f"✏️ Input Performance Indicator")
inputs = []
cols = st.columns(2)

#  
default_values = [
    0.339067,   # Proportion of Medical Service Revenue in Total Medical Revenue
    0.051967,   # Proportion of Consumables in Total Medical Revenue
    0.081767,   # Proportion of Outpatient Revenue in Total Medical Revenue
    0.438100,   # Proportion of Outpatient Revenue from Medical Insurance Fund
    0.918233,   # Proportion of Inpatient Revenue in Total Medical Revenue
    0.520033,  # Proportion of Inpatient Revenue from Medical Insurance Fund
    568.390000,  # Average Outpatient Drug Expense per Visit
    747.716667, # Average Outpatient Expense per Visit
    15032.753333, # Average Inpatient Expense per Admission
    5624.190000  # Average Inpatient Drug Expense per Admission
]

for i, feature in enumerate(features):
    with cols[i % 2]:
        val = st.number_input(
            feature, 
            min_value=0.0, 
            step=0.01, 
            format="%.2f",
            value=default_values[i]   
        )
        inputs.append(val)

# ---------------- Helper: load/prepare background for SHAP ----------------
def prepare_background(shap_bg_path, fallback_inputs, n_tile=50):
    if shap_bg_path and os.path.exists(shap_bg_path):
        try:
            df = pd.read_csv(shap_bg_path, encoding="utf-8-sig")
            if set(features).issubset(df.columns):
                return df[features].astype(float)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= len(features):
                return df[num_cols[:len(features)]].astype(float).reset_index(drop=True)
        except Exception:
            pass
    try:
        bg = pd.DataFrame(np.tile(fallback_inputs, (n_tile, 1)), columns=features)
        return bg.astype(float)
    except Exception:
        return None

# ---------------- Prediction & SHAP ----------------
if st.button("🚀 Predict Performance"):
    X_input_df = pd.DataFrame([inputs], columns=features)
    try:
        y_pred = model.predict(X_input_df)
        score = float(y_pred[0])
        grade = map_to_grade(score, quartile_ranges)

        st.success(f"✅ Predicted Annual Performance Score = **{score:.4f}**")
        st.info(f"📍 Department: {department_choice} ({department_category}) | Month Window: {window_choice}")
        st.subheader("🏷️ Corresponding Performance Grade")
        st.markdown(f"****{grade}****  Level")

        st.subheader("💡 SHAP Waterfall Plot")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 96

        X_background = prepare_background(shap_bg_path, inputs, n_tile=60)
        if X_background is None or X_background.shape[1] != len(features):
            st.warning(" ")
            X_background = pd.DataFrame(np.tile(inputs, (60, 1)), columns=features)

        try:
            explainer = shap.Explainer(model, X_background)
        except Exception:
            explainer = shap.Explainer(model.predict, X_background)

        try:
            shap_values = explainer(X_input_df)
        except Exception as e:
            st.warning(f" {e}")
            shap_values = explainer(X_background)
            shap_values = shap_values[0:1]

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            fig = plt.gcf()
        except Exception as err_primary:
            st.warning(f"  {err_primary}")
            try:
                vals = np.array(shap_values.values, dtype=float)
                base = np.array(shap_values.base_values, dtype=float)
            except Exception:
                vals = np.zeros((1, len(features)), dtype=float)
                base = np.array([float(score)])
            vals = np.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
            base = np.nan_to_num(base, nan=0.0, posinf=1e6, neginf=-1e6)
            clip_val = 1e6
            vals = np.clip(vals, -clip_val, clip_val)
            base = np.clip(base, -clip_val, clip_val)
            try:
                safe_expl = shap.Explanation(values=vals, base_values=base, data=X_input_df.values, feature_names=features)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(safe_expl[0], show=False)
                fig = plt.gcf()
            except Exception as err_secondary:
                st.error(f" {err_secondary}")
                fig = None

        if fig is not None:
            st.pyplot(fig)
            st.caption(" ")
        else:
            st.info(" ")

    except Exception as e:
        st.error(f" {e}")

