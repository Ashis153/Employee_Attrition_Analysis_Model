import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="HR Retention Strategy Dashboard", layout="wide")


@st.cache_resource
def load_all():
    # Ensure these 8 files are in your PyCharm project folder
    model = pickle.load(open('attrition_model.pkl', 'rb'))
    eltv_model = pickle.load(open('eltv_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    clf_features = pickle.load(open('clf_features.pkl', 'rb'))
    eltv_features = pickle.load(open('eltv_features.pkl', 'rb'))
    q75 = pickle.load(open('eltv_q75.pkl', 'rb'))
    q90 = pickle.load(open('eltv_q90.pkl', 'rb'))
    options = pickle.load(open('cat_options.pkl', 'rb'))
    return model, eltv_model, scaler, clf_features, eltv_features, q75, q90, options


model, eltv_model, scaler, clf_features, eltv_features, eltv_q75, eltv_q90, options = load_all()

st.title("Employee Dual-Model ML System")
st.markdown("Predicting **Attrition Risk** and **Employee Lifetime Value (ELTV)** for Data-Driven Retention.")

st.sidebar.header("Employee Profile Input")

# Group 1: Personal & Demographic
with st.sidebar.expander("Personal & Demographic", expanded=True):
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", options['Gender'])
    marital = st.selectbox("Marital Status", options['MaritalStatus'])
    education = st.slider("Education Level (1-5)", 1, 5, 3)
    edu_field = st.selectbox("Education Field", options['EducationField'])
    dist = st.slider("Distance From Home (km)", 1, 30, 5)

# Group 2: Work Experience & Role
with st.sidebar.expander("Work Experience & Role", expanded=False):
    dept = st.selectbox("Department", options['Department'])
    role = st.selectbox("Job Role", options['JobRole'])
    job_level = st.slider("Job Level", 1, 5, 2)
    income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
    travel = st.selectbox("Business Travel", options['BusinessTravel'])
    ot = st.selectbox("OverTime", options['OverTime'])
    num_comp = st.slider("Number of Companies Worked", 0, 9, 1)
    total_years = st.slider("Total Working Years", 0, 40, 10)
    stock = st.slider("Stock Option Level", 0, 3, 0)

# Group 3: Satisfaction & Performance
with st.sidebar.expander("Satisfaction & Performance", expanded=False):
    env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    job_inv = st.slider("Job Involvement (1-4)", 1, 4, 3)
    job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    rel_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
    wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    perf = st.selectbox("Performance Rating", [3, 4])
    hike = st.slider("Percent Salary Hike", 0, 25, 12)
    training = st.slider("Training Times Last Year", 0, 6, 2)

# Group 4: Tenure
with st.sidebar.expander("Tenure at Company", expanded=False):
    years_at_co = st.slider("Years at Company", 0, 40, 5)
    years_role = st.slider("Years in Current Role", 0, 20, 2)
    years_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_manager = st.slider("Years With Current Manager", 0, 20, 2)

if st.button("Generate Strategic Analysis"):
    raw_data = {
        'Age': age, 'BusinessTravel': travel, 'Department': dept, 'DistanceFromHome': dist,
        'Education': education, 'EducationField': edu_field, 'EnvironmentSatisfaction': env_sat,
        'Gender': gender, 'JobInvolvement': job_inv, 'JobLevel': job_level, 'JobRole': role,
        'JobSatisfaction': job_sat, 'MaritalStatus': marital, 'MonthlyIncome': income,
        'NumCompaniesWorked': num_comp, 'OverTime': ot, 'PercentSalaryHike': hike,
        'PerformanceRating': perf, 'RelationshipSatisfaction': rel_sat, 'StockOptionLevel': stock,
        'TotalWorkingYears': total_years, 'TrainingTimesLastYear': training, 'WorkLifeBalance': wlb,
        'YearsAtCompany': years_at_co, 'YearsInCurrentRole': years_role,
        'YearsSinceLastPromotion': years_promo, 'YearsWithCurrManager': years_manager
    }

    st.write("### Captured Employee Data")
    st.dataframe(pd.DataFrame([raw_data]))

    input_df = pd.DataFrame([raw_data])
    input_encoded = pd.get_dummies(input_df)

    scaler_input = input_encoded.reindex(columns=clf_features, fill_value=0)
    scaled_data = scaler.transform(scaler_input)

    risk_proba = model.predict_proba(scaled_data)[0][1]

    final_eltv_input = input_encoded.reindex(columns=eltv_features, fill_value=0)
    eltv_pred = eltv_model.predict(final_eltv_input)[0]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Risk Assessment")
        is_risk = "YES" if risk_proba >= 0.26 else "NO"
        risk_color = "red" if is_risk == "YES" else "green"
        st.markdown(f"### Flight Risk: :{risk_color}[{is_risk}]")
        st.write(f"Risk Probability: **{risk_proba:.1%}**")

    with col2:
        st.subheader("Economic Value Assessment")
        st.markdown(f"### Predicted ELTV: :blue[${eltv_pred:,.2f}]")
        val_status = "High Value" if eltv_pred > eltv_q75 else "Standard Value"
        st.write(f"Category: **{val_status}**")

    st.subheader("Recommended HR Strategy")

    if risk_proba >= 0.26 and eltv_pred >= eltv_q90:
        st.error(
            "💎 **CRITICAL ASSET INTERVENTION:** High-value employee at high risk. Recommend immediate retention bonus, stock options, and a career roadmap discussion.")
    elif risk_proba >= 0.26 and eltv_pred >= eltv_q75:
        st.warning(
            "⚠️ **STRATEGIC RETENTION:** Above-average value with high risk. Schedule a 'Stay Interview' and review project alignment.")
    elif risk_proba >= 0.26:
        st.info(
            "📉 **OPERATIONAL MONITORING:** High risk of leaving. Review role fit and provide standard engagement feedback.")
    else:
        st.success("✅ **STABLE & GROWING:** Low risk. Continue standard performance rewards and engagement.")


