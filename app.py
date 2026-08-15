import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

st.title("📊 Thesis Helper: Advanced Business Analytics")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.write("### Data Preview", df.head())

    # Sidebar Menu
    analysis_type = st.sidebar.selectbox("Choose Analysis Type", 
                                         ["Descriptive Statistics", "Correlation Analysis", "Linear Regression", "T-Test"])

    # 1. Descriptive
    if analysis_type == "Descriptive Statistics":
        st.write("### Descriptive Statistics", df.describe())

    # 2. Correlation
    elif analysis_type == "Correlation Analysis":
        st.write("### Correlation Matrix")
        corr = df.select_dtypes(include='number').corr()
        st.write(corr)
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # 3. Regression
    elif analysis_type == "Linear Regression":
        target = st.selectbox("Select Dependent Variable (Y)", df.columns)
        features = st.multiselect("Select Independent Variables (X)", df.columns)
        if st.button("Run Regression"):
            X = df[features]
            y = df[target]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            st.write(model.summary())

    # 4. T-Test
    elif analysis_type == "T-Test":
        group_col = st.selectbox("Select Categorical Column", df.columns)
        val_col = st.selectbox("Select Numerical Column", df.select_dtypes(include='number').columns)
        if st.button("Run T-Test"):
            groups = df[group_col].unique()
            data1 = df[df[group_col] == groups[0]][val_col]
            data2 = df[df[group_col] == groups[1]][val_col]
            t_stat, p_val = stats.ttest_ind(data1, data2)
            st.write(f"T-statistic: {t_stat}, P-value: {p_val}")

else:
    st.info("Please upload a file to start the analysis.")
