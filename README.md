<div align="center">

# XAI-Powered Heart Disease Prediction System  
### Clinical Decision Support with Explainable Artificial Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clinically interpretable machine learning system for heart disease prediction with explainable AI using SHAP.

</div>

---

## 1. Project Overview

This project implements an Explainable AI (XAI) based heart disease prediction system designed for clinical decision support. The system predicts heart disease risk using Logistic Regression and provides interpretable explanations using SHAP (SHapley Additive exPlanations).

The goal is not only high predictive performance but also transparency, stability, and clinical trust.

---

## 2. Core Objectives

1. Build a reliable heart disease prediction model  
2. Provide transparent explanations for each prediction  
3. Validate explanation faithfulness and stability  
4. Deliver a clean, user-friendly clinical interface  
5. Enable medical interpretability rather than black-box output  

---

## 3. Why Logistic Regression?

Although tree-based models were evaluated, Logistic Regression was selected because:

| Criterion | Logistic Regression | Random Forest |
|------------|-------------------|---------------|
| Cross-Validated ROC-AUC | Comparable | Comparable |
| Overfitting | Low | High (AUC = 1.0 training) |
| Faithfulness | Strong | Moderate |
| Stability | High (~0.71) | Very Low (~0.20) |
| Interpretability | Intrinsic | Post-hoc |

For medical AI systems, stability and interpretability are more critical than marginal accuracy gains.

---

## 4. System Architecture


User Input (Streamlit UI)
↓
Data Preprocessing (Scaler)
↓
Logistic Regression Model
↓
Probability Output
↓
SHAP Explainability Engine
↓
Clinical Interpretation Layer


---

## 5. Features

1. Binary heart disease prediction (High / Low Risk)
2. Probability score output
3. SHAP waterfall visualization
4. Personalized clinical explanation
5. Risk stratification (Low / Moderate / High)
6. Faithfulness validation
7. Stability validation
8. Clean Streamlit interface
9. GitHub-based version control
10. Deployable architecture

---

## 6. Technology Stack

Backend:
- Python 3.9+
- Scikit-learn
- SHAP
- NumPy
- Pandas

Frontend:
- Streamlit

Deployment:
- GitHub
- Git LFS (for model files)
- Streamlit Cloud (optional)

---

## 7. Project Structure


XAI-Heart-Disease-Prediction/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md
├── data/
│ └── raw/
├── models/
│ ├── logistic_model.pkl
│ ├── scaler.pkl
│ ├── metadata.pkl
│ └── shap_explainer.pkl
├── notebooks/
│ └── 01_eda_clean.ipynb
└── .gitignore


---

## 8. Installation Guide

### Step 1: Clone the repository

```bash
git clone https://github.com/kartikpagariya25/XAI-Heart-Disease-Prediction.git
cd XAI-Heart-Disease-Prediction
Step 2: Create virtual environment (recommended)

Windows:

python -m venv venv
venv\Scripts\activate

Linux / Mac:

python -m venv venv
source venv/bin/activate
Step 3: Install dependencies
pip install -r requirements.txt
Step 4: Run the application
streamlit run app.py

Application will be available at:

http://localhost:8501
9. Usage Workflow

Enter patient clinical parameters in sidebar

Click "Predict Risk"

View:

Probability score

Risk category

SHAP waterfall chart

Professional clinical interpretation

Risk categories:

Low Risk: Probability < 0.40

Moderate Risk: 0.40–0.70

High Risk: > 0.70

10. Explainable AI Implementation
Local Explanations

SHAP waterfall plots

Feature contribution ranking

Clinical narrative explanation

Global Evaluation

Cross-validation

Faithfulness testing (feature removal impact)

Stability testing (cosine similarity of SHAP vectors)

11. Validation Metrics

Faithfulness:
Removal of top 3 features caused significant ROC-AUC drop (~0.07).

Stability:
Logistic Regression achieved moderate-to-strong stability (~0.71).

These metrics confirm explanation reliability.

12. Team Contributions
Member	Role
Kartik	ML & Explainability Lead
Aditya	UI/UX Engineer
Vikrant	DevOps & Deployment
Pranali	Research Documentation
Janhavi	Testing & Validation
13. Future Improvements

Integrate LIME for comparative XAI analysis

Add neural network models

Add downloadable PDF medical report

Integrate EHR compatibility

Add deployment monitoring

14. Dataset

Heart Disease Dataset – UCI Machine Learning Repository

15. License

MIT License

<div align="center">

Built as an academic Explainable AI system for clinical decision support.

</div> ```