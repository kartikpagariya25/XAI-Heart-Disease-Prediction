<div align="center">

# 🫀 XAI Heart Disease Prediction System

### Clinical Decision Support with Explainable AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributors](https://img.shields.io/badge/Contributors-5-orange.svg)](#team)

*A machine learning-powered heart disease prediction system with interpretable AI explanations using SHAP values.*

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Team & Tasks](#team--tasks)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [XAI Implementation](#xai-implementation)
- [Future Scope](#future-scope)

---

## 🎯 About the Project

This project implements an **Explainable AI (XAI)** powered heart disease prediction system designed for clinical decision support. The system uses a Logistic Regression model to predict heart disease risk and provides interpretable explanations through **SHAP (SHapley Additive exPlanations)** values, enabling healthcare professionals to understand the reasoning behind each prediction.

### Key Objectives

- 🔍 **Predictive Accuracy**: Build a reliable heart disease prediction model
- 📊 **Explainability**: Provide transparent, clinically interpretable predictions
- 🏥 **Clinical Relevance**: Enable doctors to validate and trust AI-assisted diagnoses
- 📈 **Risk Stratification**: Categorize patients into low, moderate, and high risk

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🫀 **Heart Disease Prediction** | Binary classification (High/Low risk) using Logistic Regression |
| 📊 **SHAP Explanations** | Global and local feature importance visualizations |
| 🎨 **Waterfall Charts** | Visual representation of individual prediction contributions |
| 📝 **Clinical Interpretations** | User-friendly explanations of risk factors |
| 🛡️ **Privacy-Focused** | All processing happens locally - no patient data leaves the system |
| 📱 **Web Interface** | Intuitive Streamlit application for easy interaction |

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                           │
│                      Streamlit Web App                          │
├─────────────────────────────────────────────────────────────────┤
│                      Backend Intelligence                       │
│              Logistic Regression + SHAP XAI Engine              │
├─────────────────────────────────────────────────────────────────┤
│                       Data Processing                           │
│                  Pandas, NumPy, Scikit-learn                    │
├─────────────────────────────────────────────────────────────────┤
│                        Deployment                               │
│                 Streamlit Cloud + GitHub                        │
└─────────────────────────────────────────────────────────────────┘
```

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Web application framework |
| **Scikit-learn** | ML model (Logistic Regression) |
| **SHAP** | Explainable AI implementation |
| **Pandas/NumPy** | Data processing |
| **Matplotlib/Seaborn** | Visualization |
| **Joblib** | Model serialization |

---

## 👥 Team & Tasks

We are a cross-functional team of 5 members working together to build an end-to-end Explainable AI system for heart disease prediction.

---

### 1️⃣ Kartik — Core ML & Explainability Lead

**Primary Responsibility:** Model and XAI Engine

| Tasks |
|-------|
| ✅ Data preprocessing pipeline design |
| ✅ Logistic Regression model training |
| ✅ Cross-validation and performance evaluation |
| ✅ SHAP implementation (global + local explanations) |
| ✅ Faithfulness metric implementation |
| ✅ Stability metric implementation |
| ✅ Model selection justification |
| ✅ Saving trained model, scaler, metadata |
| ✅ SHAP integration logic in Streamlit |
| ✅ Technical architecture decisions |

**Deliverables:**
- Final trained model
- SHAP explanation engine
- Evaluation metrics
- Model comparison documentation

**Ownership Area:** Backend Intelligence Layer

---

### 2️⃣ Aditya — Streamlit UI & UX Engineer

**Primary Responsibility:** Frontend Application

| Tasks |
|-------|
| ✅ Design structured Streamlit layout |
| ✅ Sidebar input structuring |
| ✅ Professional input tooltips |
| ✅ User-friendly explanation formatting |
| ✅ Risk categorization display |
| ✅ Layout improvements (columns, spacing, structure) |
| ✅ Error handling and input validation |
| ✅ UI refinement for demo presentation |
| ✅ Screenshots for documentation |

**Deliverables:**
- Clean, professional Streamlit interface
- Final demo-ready app.py UI layer

**Ownership Area:** User Interaction Layer

---

### 3️⃣ Vikrant — Deployment & DevOps

**Primary Responsibility:** GitHub + Deployment

| Tasks |
|-------|
| ✅ GitHub repository structuring |
| ✅ requirements.txt management |
| ✅ .gitignore configuration |
| ✅ Version control management |
| ✅ Streamlit Cloud deployment |
| ✅ Testing deployment stability |
| ✅ Managing model size constraints |
| ✅ Creating project directory structure |
| ✅ CI/CD readiness (optional future scope) |

**Deliverables:**
- Public GitHub repository
- Live deployed application link
- Clean repository structure

**Ownership Area:** Infrastructure Layer

---

### 4️⃣ Pranali — Research Documentation & Analysis

**Primary Responsibility:** Research Writing

| Tasks |
|-------|
| ✅ Literature review (XAI in healthcare) |
| ✅ Problem statement drafting |
| ✅ Methodology documentation |
| ✅ Model comparison section writing |
| ✅ Faithfulness & Stability explanation section |
| ✅ Results & discussion writing |
| ✅ Conclusion drafting |
| ✅ Limitations & future scope section |
| ✅ Preparing PPT content |
| ✅ Preparing research paper draft |

**Deliverables:**
- Research paper draft
- Project report
- Presentation slides

**Ownership Area:** Academic Documentation

---

### 5️⃣ Janhavi — Testing, Validation & Clinical Interpretation

**Primary Responsibility:** Quality Assurance & Validation

| Tasks |
|-------|
| ✅ Test case generation |
| ✅ Edge case testing |
| ✅ Clinical consistency validation |
| ✅ Comparing SHAP outputs with medical logic |
| ✅ Testing low, moderate, high risk cases |
| ✅ Debugging explanation mismatches |
| ✅ Validating input mapping correctness |
| ✅ Creating demo scenarios |
| ✅ Writing clinical interpretation validation notes |

**Deliverables:**
- Testing report
- Validation documentation
- Demo case booklet

**Ownership Area:** Validation & Reliability Layer

---

## 👨‍💻 Team Members

| Member | Role | GitHub |
|--------|------|--------|
| **Kartik** | Core ML & XAI Lead | [kartikpagariya25](https://github.com/kartikpagariya25) |
| **Aditya** | UI/UX Engineer | [DevXDividends](https://github.com/DevXDividends) |
| **Vikrant** | DevOps & Deployment | [VikrantKadam028](https://github.com/VikrantKadam028) |
| **Pranali** | Research Documentation | - |
| **Janhavi** | Testing & Validation | - |

---

## 📂 Project Structure

```
XAI/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── raw/                    # Raw heart disease dataset
│   └── processed/              # Preprocessed data
├── models/                     # Trained models and artifacts
│   ├── logistic_model.pkl       # Trained Logistic Regression
│   ├── scaler.pkl               # Feature scaler
│   ├── metadata.pkl            # Model metadata
│   └── shap_explainer.pkl      # SHAP explainer
├── notebooks/
│   └── 01_eda_clean.ipynb      # EDA and data cleaning
└── PROJECT_DOCUMENTATION.md    # Detailed project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Steps

1. **Clone the repository**
   
```
bash
   git clone https://github.com/your-repo/xai-heart-disease.git
   cd xai-heart-disease
   
```

2. **Create a virtual environment (optional but recommended)**
   
```
bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
```

3. **Install dependencies**
   
```
bash
   pip install -r requirements.txt
   
```

4. **Run the application**
   
```
bash
   streamlit run app.py
   
```

5. **Open in browser**
   Navigate to `http://localhost:8501` in your web browser

---

## 📖 Usage

### Using the Web Application

1. **Enter Patient Data**: Use the sidebar to input clinical parameters:
   - Age, Sex, Chest Pain Type
   - Blood Pressure, Cholesterol
   - Fasting Blood Sugar, Resting ECG
   - Maximum Heart Rate, Exercise Angina
   - ST Depression, Slope, Vessels, Thalassemia

2. **Click "Predict Risk"**: Get instant prediction with probability score

3. **View Explanations**:
   - 📊 **Waterfall Chart**: Visual breakdown of feature contributions
   - 📝 **Clinical Interpretation**: Human-readable risk factors
   - 🛡️ **Protective Factors**: What reduces the risk

4. **Risk Categories**:
   - 🟢 **Low Risk**: Probability < 40%
   - 🟡 **Moderate Risk**: Probability 40-70%
   - 🔴 **High Risk**: Probability > 70%

---

## 🔬 Model Details

### Algorithm
- **Logistic Regression** with L2 regularization

### Performance Metrics
- Cross-validation for robust evaluation
- Balanced accuracy, Precision, Recall, F1-Score

### Features Used
| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Gender (0: Female, 1: Male) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar >120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia test result |

---

## 🧠 XAI Implementation

### SHAP (SHapley Additive exPlanations)

Our XAI implementation provides:

1. **Local Explanations**: Understanding individual predictions
   - Waterfall charts showing feature contributions
   - Personalized clinical interpretation

2. **Global Explanations**: Understanding model behavior
   - Feature importance rankings
   - Feature interaction analysis

### Metrics Implemented

- **Faithfulness**: How well explanations correlate with model behavior
- **Stability**: Consistency of explanations for similar cases

---

## 🔮 Future Scope

- [ ] Add more XAI methods (LIME, Integrated Gradients)
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add time-series analysis for patient history
- [ ] Integrate with Electronic Health Records (EHR)
- [ ] Mobile application development
- [ ] Real-time patient monitoring integration
- [ ] Multi-class classification (different heart conditions)

---


## 🙏 Acknowledgments

- Heart Disease Dataset: UCI Machine Learning Repository
- SHAP: Scott M. Lundberg et al.
- Streamlit for the amazing web framework

---

<div align="center">

**Built with ❤️ by Team XAI**

*Empowering Healthcare with Explainable AI*

</div>
