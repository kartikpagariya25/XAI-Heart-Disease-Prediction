import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.special import expit

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="XAI Heart Disease Prediction",
    layout="wide"
)

st.title("XAI Powered Heart Disease Prediction System")
st.markdown("Clinical Decision Support with Explainable AI")

# ======================================================
# Load Artifacts
# ======================================================

model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")
metadata = joblib.load("models/metadata.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

feature_order = metadata["feature_order"]
numerical_features = metadata["numerical_features"]

# ======================================================
# Sidebar Inputs
# ======================================================

st.sidebar.header("Patient Clinical Parameters")
st.sidebar.markdown("Provide the patient's clinical details below.")

age = st.sidebar.slider("Age (Years)", 20, 100, 50)

sex_label = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0

cp_label = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_mapping[cp_label]

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs_label = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No", "Yes"])
fbs = 1 if fbs_label == "Yes" else 0

restecg_label = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)
restecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_mapping[restecg_label]

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exang_label = st.sidebar.selectbox("Exercise Induced Angina?", ["No", "Yes"])
exang = 1 if exang_label == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

slope_label = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)
slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_mapping[slope_label]

ca = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])

thal_label = st.sidebar.selectbox(
    "Thalassemia Test Result",
    ["Normal", "Fixed Defect", "Reversible Defect", "Other Abnormality"]
)
thal_mapping = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
    "Other Abnormality": 3
}
thal = thal_mapping[thal_label]

# ======================================================
# Prediction Section
# ======================================================

if st.button("Predict Risk"):

    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    # Arrange features in correct order
    input_array = np.array([input_dict[col] for col in feature_order]).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=feature_order)

    # Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prob > 0.5:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    st.write(f"Predicted Probability: {prob:.3f}")

    # ======================================================
    # SHAP Explanation
    # ======================================================

    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_vector = shap_values[0]
    base_value = explainer.expected_value

    st.subheader("Visual Explanation")

    explanation = shap.Explanation(
        values=shap_vector,
        base_values=base_value,
        data=input_df.iloc[0],
        feature_names=feature_order
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # ======================================================
    # User-Friendly Explanation
    # ======================================================

    st.subheader("Personalized Risk Explanation")

    # ======================================================
    # Professional Clinical Explanation
    # ======================================================

    st.subheader("Personalized Clinical Interpretation")

    contribution_df = pd.DataFrame({
        "Feature": feature_order,
        "Impact": shap_vector
    }).sort_values(by="Impact", key=abs, ascending=False)

    positive = contribution_df[contribution_df["Impact"] > 0].head(3)
    negative = contribution_df[contribution_df["Impact"] < 0].head(3)

    st.markdown("### Key Risk-Enhancing Factors")

    if len(positive) == 0:
        st.write("No major clinical parameters were found to significantly elevate the predicted risk.")
    else:
        for _, row in positive.iterrows():
            feature = row["Feature"]
            value = input_dict[feature]

            if feature == "age":
                st.write(
                    f"- The patient's age of {value} years increases cardiovascular risk, "
                    "as advancing age is a well-established non-modifiable risk factor."
                )

            elif feature == "oldpeak":
                st.write(
                    f"- An ST depression (Oldpeak) value of {value} suggests abnormal cardiac stress response, "
                    "which is associated with elevated risk."
                )

            elif feature == "ca":
                st.write(
                    f"- The presence of {value} major vessel(s) with blockage indicates structural coronary involvement."
                )

            elif feature == "thal":
                st.write(
                    "- The thalassemia test result indicates abnormal myocardial perfusion, "
                    "which may reflect compromised blood flow."
                )

            elif feature == "exang":
                st.write(
                    "- Exercise-induced angina suggests that physical stress provokes cardiac symptoms, "
                    "a clinically relevant risk indicator."
                )

            elif feature == "trestbps":
                st.write(
                    f"- A resting blood pressure of {value} mm Hg contributes to increased cardiac workload."
                )

            elif feature == "chol":
                st.write(
                    f"- A cholesterol level of {value} mg/dl may contribute to arterial plaque formation."
                )

            elif feature == "fbs":
                st.write(
                    "- Elevated fasting blood sugar suggests possible metabolic imbalance, "
                    "which is linked to cardiovascular complications."
                )

            elif feature == "slope":
                st.write(
                    "- The observed ST segment slope pattern is associated with higher cardiac stress."
                )

            elif feature == "sex":
                st.write(
                    "- Male gender is statistically associated with increased prevalence of heart disease."
                )

            else:
                st.write(f"- {feature} is contributing to elevated predicted risk.")

    st.markdown("### Key Protective Factors")

    if len(negative) == 0:
        st.write("No strong protective factors were identified.")
    else:
        for _, row in negative.iterrows():
            feature = row["Feature"]
            value = input_dict[feature]

            if feature == "thalach":
                st.write(
                    f"- A maximum heart rate of {value} bpm indicates good exercise tolerance, "
                    "which is generally associated with better cardiac fitness."
                )

            elif feature == "cp":
                st.write(
                    "- The reported chest pain pattern does not correspond to high-risk angina."
                )

            elif feature == "ca":
                st.write(
                    "- The absence of major vessel blockage reduces structural cardiac risk."
                )

            elif feature == "thal":
                st.write(
                    "- The thalassemia result does not indicate significant perfusion abnormality."
                )

            elif feature == "oldpeak":
                st.write(
                    "- Minimal ST depression suggests normal cardiac response to stress."
                )

            elif feature == "trestbps":
                st.write(
                    "- Resting blood pressure appears within a relatively controlled range."
                )

            elif feature == "chol":
                st.write(
                    "- Cholesterol level does not indicate severe lipid-related risk."
                )

            elif feature == "age":
                st.write(
                    "- Younger age reduces overall baseline cardiovascular risk."
                )

            else:
                st.write(f"- {feature} is contributing to reduced predicted risk.")

    st.markdown("### Overall Clinical Assessment")

    if prob > 0.7:
        st.write(
            "The predicted probability indicates a high likelihood of heart disease. "
            "Multiple clinically significant risk factors are present. "
            "Comprehensive cardiovascular evaluation is strongly recommended."
        )
    elif prob > 0.4:
        st.write(
            "The predicted probability indicates moderate cardiovascular risk. "
            "Some clinical parameters warrant further diagnostic screening."
        )
    else:
        st.write(
            "The predicted probability indicates low cardiovascular risk. "
            "Most evaluated clinical indicators appear within safer physiological ranges."
        )