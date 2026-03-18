#  📉 Employee_Attrition_Analysis_Model
![Model Output](https://github.com/Ashis153/Employee_Attrition_Analysis_Model/blob/main/Screenshot%202026-02-23%20224704.png)


This high-impact **Human Capital Management (HCM) analytics suite** utilizes a dual-model architecture to solve two core business challenges: predicting **Employee Attrition (Flight Risk)** and estimating **Employee Lifetime Value (ELTV)**.

The system provides HR leaders with actionable insights, identifying not just *who* might leave, but the *economic impact* of that loss to prioritize high-value interventions.

### 🔗 Live Demo
> **Access the interactive dashboard here:** [👉 Employee Dual-Model ML System](https://your-link-here.streamlit.app)

---

## 🏗️ Dual-Model Architecture
The core of the system relies on two specialized pipelines designed to handle the inherent imbalance and complexity of HR data:

### 1. Attrition Classification (Flight Risk)
* **Objective:** Predict the probability of an employee leaving the organization.
* **Handling Imbalance:** Utilized **SMOTE** (*Synthetic Minority Over-sampling Technique*) via `imblearn` to address the class imbalance between "Stayed" vs "Left" employees.
* **Algorithms:** Tuned `Random Forest Classifier` and `XGBoost` pipeline optimized for **High Recall**.
* **Thresholding:** Applied a custom classification threshold of $0.26$ to prioritize proactive intervention over raw accuracy.

### 2. ELTV Regression (Economic Value)
* **Objective:** Forecast the projected financial or utility contribution of an employee.
* **Algorithm:** `Random Forest Regressor` to capture non-linear relationships between features like *Monthly Income*, *Job Level*, and *Performance Rating*.
* **Strategic Segmentation:** Predictions are mapped against $Q_{75}$ and $Q_{90}$ quartiles to identify "High Value" and "Critical" talent.

---

## 🚀 Key Features

* **What-If Simulation:** Interactive Streamlit sidebar allows users to adjust parameters (Overtime, Income, Satisfaction) to see real-time shifts in risk and value.
* **Automated Strategy Engine:** Categorizes employees into action-oriented tiers:
    * 🔴 **Critical Asset Interventions:** High Risk + High ELTV
    * 🟡 **Strategic Retention:** Medium Risk + High ELTV
    * 🟢 **Standard Monitoring:** Low Risk

---

## 🛠️ Technical Stack

| Domain | Technologies |
| :--- | :--- |
| **Data Science** | `Pandas`, `NumPy`, `Seaborn`, `Matplotlib` |
| **Machine Learning** | `Scikit-Learn`, `Imbalanced-Learn (SMOTE)`, `XGBoost` |
| **Deployment** | `Streamlit`, `Pickle` (Serialization) |

---

## 📊 Methodology

1.  **Exploratory Data Analysis (EDA):** Identified key drivers such as *Work-Life Balance* and *Stock Option Levels*.
2.  **Feature Engineering:** Categorical encoding and standard scaling to prepare raw HR data.
3.  **Threshold Optimization:** Custom logic to maximize recall for flight risk detection.
4.  **Deployment:** Modularized `main.py` that loads pre-trained weights and scalers for instant inference.
