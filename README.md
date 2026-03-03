<div align="center">

<h1>XAI-Powered Heart Disease Prediction System</h1>
<h3>Clinical Decision Support with Explainable Artificial Intelligence</h3>

<p>
  <img src="https://img.shields.io/github/stars/kartikpagariya25/XAI-Heart-Disease-Prediction?style=social" />
  <img src="https://img.shields.io/github/forks/kartikpagariya25/XAI-Heart-Disease-Prediction?style=social" />
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" />
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-orange.svg" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-critical.svg" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg" />
  <img src="https://img.shields.io/badge/Deployment-GitHub-black.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

<p>
  <img src="https://img.shields.io/github/last-commit/kartikpagariya25/XAI-Heart-Disease-Prediction" />
  <img src="https://img.shields.io/github/issues/kartikpagariya25/XAI-Heart-Disease-Prediction" />
  <img src="https://img.shields.io/github/languages/top/kartikpagariya25/XAI-Heart-Disease-Prediction" />
</p>

<p>
A clinically interpretable machine learning system for heart disease prediction
with transparent AI explanations using SHAP.
</p>

</div>

<hr/>

<h2>1. Project Overview</h2>

<p>
This project implements an Explainable AI (XAI) powered heart disease prediction system 
designed for clinical decision support. The system uses Logistic Regression for prediction 
and SHAP (SHapley Additive exPlanations) to generate interpretable explanations for each case.
</p>

<p>
The primary objective is to balance predictive performance with explanation stability 
and clinical interpretability.
</p>

<hr/>

<h2>2. Core Objectives</h2>

<ol>
  <li>Develop a reliable heart disease risk prediction model</li>
  <li>Provide transparent and faithful AI explanations</li>
  <li>Validate explanation stability across similar cases</li>
  <li>Deliver a professional web interface for clinicians</li>
  <li>Ensure structured version control and deployment</li>
</ol>

<hr/>

<h2>3. Model Selection Justification</h2>

<table>
<tr>
<th>Criterion</th>
<th>Logistic Regression</th>
<th>Random Forest</th>
</tr>
<tr>
<td>Cross-Validated ROC-AUC</td>
<td>Comparable</td>
<td>Comparable</td>
</tr>
<tr>
<td>Overfitting</td>
<td>Low</td>
<td>High (Training AUC = 1.0)</td>
</tr>
<tr>
<td>Faithfulness</td>
<td>Strong</td>
<td>Moderate</td>
</tr>
<tr>
<td>Stability</td>
<td>High (~0.71)</td>
<td>Very Low (~0.20)</td>
</tr>
<tr>
<td>Interpretability</td>
<td>Intrinsic</td>
<td>Post-hoc</td>
</tr>
</table>

<p>
Logistic Regression was selected due to superior explanation stability and intrinsic interpretability,
which are critical for medical AI systems.
</p>

<hr/>

<h2>4. System Architecture</h2>

<pre>
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
</pre>

<hr/>

<h2>5. Key Features</h2>

<ul>
  <li>Binary heart disease prediction (High / Low Risk)</li>
  <li>Probability-based risk scoring</li>
  <li>SHAP waterfall visualizations</li>
  <li>Personalized clinical explanation</li>
  <li>Faithfulness and stability validation</li>
  <li>Git LFS model management</li>
  <li>Professional Streamlit interface</li>
</ul>

<hr/>

<h2>6. Technology Stack</h2>

<ul>
  <li><b>Backend:</b> Python, Scikit-learn, SHAP, Pandas, NumPy</li>
  <li><b>Frontend:</b> Streamlit</li>
  <li><b>Deployment:</b> GitHub + Git LFS</li>
</ul>

<hr/>

<h2>7. Installation</h2>

<pre>
git clone https://github.com/kartikpagariya25/XAI-Heart-Disease-Prediction.git
cd XAI-Heart-Disease-Prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
</pre>

Application runs at: <b>http://localhost:8501</b>

<hr/>

<h2>8. Risk Stratification</h2>

<ul>
  <li><b>Low Risk:</b> Probability &lt; 0.40</li>
  <li><b>Moderate Risk:</b> 0.40 – 0.70</li>
  <li><b>High Risk:</b> &gt; 0.70</li>
</ul>

<hr/>

<h2>9. Explainable AI Validation</h2>

<ul>
  <li><b>Faithfulness:</b> Removing top features caused ~0.07 ROC-AUC drop</li>
  <li><b>Stability:</b> Logistic model achieved ~0.71 cosine similarity</li>
  <li><b>Clinical Alignment:</b> SHAP explanations aligned with known medical risk factors</li>
</ul>

<hr/>

<h2>10. Team & GitHub Profiles</h2>

<table>
<tr>
<th>Member</th>
<th>Role</th>
<th>GitHub Profile</th>
</tr>

<tr>
<td>Kartik</td>
<td>ML & Explainability Lead</td>
<td>
<a href="https://github.com/kartikpagariya25">
<img src="https://img.shields.io/badge/GitHub-kartikpagariya25-black?logo=github" />
</a>
</td>
</tr>

<tr>
<td>Aditya</td>
<td>UI/UX Engineer</td>
<td>
<a href="https://github.com/DevXDividends">
<img src="https://img.shields.io/badge/GitHub-DevXDividends-black?logo=github" />
</a>
</td>
</tr>

<tr>
<td>Vikrant</td>
<td>DevOps & Deployment</td>
<td>
<a href="https://github.com/VikrantKadam028">
<img src="https://img.shields.io/badge/GitHub-VikrantKadam028-black?logo=github" />
</a>
</td>
</tr>

<tr>
<td>Pranali</td>
<td>Research Documentation</td>
<td>-</td>
</tr>

<tr>
<td>Janhavi</td>
<td>Testing & Validation</td>
<td>-</td>
</tr>

</table>

<hr/>

<h2>11. Future Scope</h2>

<ol>
  <li>Integration of additional XAI methods (LIME, Integrated Gradients)</li>
  <li>Neural network-based comparative analysis</li>
  <li>Downloadable PDF clinical reports</li>
  <li>EHR system integration</li>
  <li>Production monitoring dashboard</li>
</ol>

<hr/>

<div align="center">
<b>Built as an academic Explainable AI system for clinical decision support.</b>
</div>
