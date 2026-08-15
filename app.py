Import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Survey Thesis Analyzer Pro", layout="wide")

st.title("🎓 Survey Dataset AI Thesis & Statistical Analysis Pro")
st.markdown("မြန်မာနိုင်ငံဆိုင်ရာ Survey ဒေတာများအတွက် သီးသန့်ပြုပြင်ထားသော အလိုအလျောက် ဒေတာသန့်စင်ခြင်း (Cleaning) နှင့် စာရင်းအင်းစစ်ဆေးခြင်း စနစ်။")

# Helper functions for data cleaning
def burmese_to_english_digits(text):
    if isinstance(text, str):
        burmese_digits = "၀၁၂၃၄၅၆၇၈၉"
        english_digits = "0123456789"
        trans_table = str.maketrans(burmese_digits, english_digits)
        return text.translate(trans_table)
    return text

def extract_number(val):
    if pd.isna(val):
        return np.nan
    val_str = burmese_to_english_digits(str(val))
    digits = re.findall(r'\d+', val_str)
    if digits:
        return float("".join(digits))
    return np.nan

uploaded_file = st.file_uploader("📊 Survey Excel (.xlsx) သို့မဟုတ် CSV ဖိုင် တင်ပါ", type=["csv", "xlsx"])

if uploaded_file:
    # 1. Load Data
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    
    st.info(f"📥 မူရင်းဒေတာ အတန်းရေ: {len(df)} | ကော်လံရေ: {len(df.columns)}")
    
    # 2. Advanced Survey Data Cleaning
    df_clean = df.copy()
    
    # Clean specific survey columns that contain mixed text/numbers
    # Age column (ကော်လံ ၂)
    age_col = df_clean.columns[2]
    df_clean[age_col] = df_clean[age_col].apply(extract_number)
    
    # Cost/Expense columns (ကော်လံ ၁၄ နှင့် ၁၅)
    for col_idx in [14, 15]:
        if col_idx < len(df_clean.columns):
            c_name = df_clean.columns[col_idx]
            df_clean[c_name] = df_clean[c_name].apply(extract_number)

    # Clean other text-numeric columns if needed
    for col in df_clean.columns:
        # If column looks like it should be numeric (like hours or scale values)
        if df_clean[col].dtype == 'object':
            # Try converting text numbers
            converted = df_clean[col].apply(extract_number)
            # If a good portion of values successfully converted, make it numeric
            if converted.notnull().mean() > 0.5:
                df_clean[col] = converted

    # Drop rows where critical numeric columns are NaN, or fillna
    df_clean = df_clean.dropna(subset=[age_col])
    
    st.success(f"✨ ဒေတာသန့်စင်ခြင်း ပြီးစီးပါပြီ။ (အသုံးပြုနိုင်သော အတန်းရေ: {len(df_clean)})")
    
    num_cols = df_clean.select_dtypes(include='number').columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Sidebar Navigation
    st.sidebar.header("⚙️ Analysis Modules")
    module = st.sidebar.selectbox("Choose Module", [
        "1. Cleaned Data & Descriptive Stats",
        "2. Regression Models (Simple, Multiple, Logistic)",
        "3. Goodness of Fit & Diagnostics",
        "4. Comparative Tests (T-Test, ANOVA)",
        "5. AI Executive Summary & Report"
    ])
    
    if module == "1. Cleaned Data & Descriptive Stats":
        st.subheader("📋 Cleaned Dataset Preview")
        st.dataframe(df_clean.head())
        st.markdown("#### Numerical Variables Summary")
        st.dataframe(df_clean[num_cols].describe())
        
        if len(num_cols) >= 2:
            st.markdown("#### Correlation Matrix")
            corr = df_clean[num_cols].corr()
            st.dataframe(corr)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif module == "2. Regression Models (Simple, Multiple, Logistic)":
        st.subheader("📈 Regression Analysis Suite")
        reg_type = st.selectbox("Select Regression Type", ["Simple Linear Regression", "Multiple Linear Regression", "Logistic Regression (Binary)"])
        
        if reg_type == "Simple Linear Regression":
            if len(num_cols) >= 2:
                y_var = st.selectbox("Dependent Variable (Y)", num_cols)
                x_var = st.selectbox("Independent Variable (X)", [c for c in num_cols if c != y_var])
                if st.button("Run Simple Regression"):
                    sub_df = df_clean[[y_var, x_var]].dropna()
                    model = smf.ols(f"Q('{y_var}') ~ Q('{x_var}')", data=sub_df).fit()
                    st.text(str(model.summary()))
            else:
                st.warning("ဂဏန်းကော်လံ ၂ ခု လိုအပ်ပါသည်။")
                
        elif reg_type == "Multiple Linear Regression":
            if len(num_cols) >= 3:
                y_var = st.selectbox("Dependent Variable (Y)", num_cols)
                x_vars = st.multiselect("Independent Variables (X's)", [c for c in num_cols if c != y_var])
                if x_vars and st.button("Run Multiple Regression"):
                    formula = f"Q('{y_var}') ~ " + " + ".join([f"Q('{x}')" for x in x_vars])
                    sub_df = df_clean[[y_var] + x_vars].dropna()
                    model = smf.ols(formula, data=sub_df).fit()
                    st.text(str(model.summary()))
            else:
                st.warning("Multiple Regression အတွက် ဂဏန်းကော်လံ အနည်းဆုံး ၃ ခု လိုအပ်ပါသည်။")
                
        elif reg_type == "Logistic Regression (Binary)":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                y_var = st.selectbox("Binary Dependent Variable (Target Group)", cat_cols)
                x_vars = st.multiselect("Predictors (X's)", num_cols)
                if x_vars and st.button("Run Logistic Regression"):
                    df_log = df_clean[[y_var] + x_vars].dropna()
                    target_vals = df_log[y_var].unique()
                    if len(target_vals) == 2:
                        df_log['target_encoded'] = (df_log[y_var] == target_vals[0]).astype(int)
                        formula = f"target_encoded ~ " + " + ".join([f"Q('{x}')" for x in x_vars])
                        logit_model = smf.logit(formula, data=df_log).fit()
                        st.text(str(logit_model.summary()))
                    else:
                        st.error("Target variable တွင် အုပ်စု ၂ ခုသာ ရှိရပါမည်။")
            else:
                st.warning("Logistic Regression အတွက် Categorical Dependent တစ်ခုနှင့် Numerical Predictors များ လိုအပ်ပါသည်။")

    elif module == "3. Goodness of Fit & Diagnostics":
        st.subheader("🔍 Model Diagnostics & Goodness of Fit Tests")
        if len(num_cols) >= 2:
            y_var = st.selectbox("Dependent Variable (Y)", num_cols, key="gof_y")
            x_vars = st.multiselect("Independent Variables (X's)", [c for c in num_cols if c != y_var], key="gof_x")
            
            if x_vars and st.button("Run Diagnostic Tests"):
                sub_df = df_clean[[y_var] + x_vars].dropna()
                formula = f"Q('{y_var}') ~ " + " + ".join([f"Q('{x}')" for x in x_vars])
                model = smf.ols(formula, data=sub_df).fit()
                
                st.markdown(f"**1. R-squared / Adjusted R-squared:** `{model.rsquared:.4f}` / `{model.rsquared_adj:.4f}`")
                st.markdown(f"**2. F-statistic & P-value:** F = `{model.fvalue:.4f}`, P = `{model.f_pvalue:.4e}`")
                
                dw = durbin_watson(model.resid)
                st.markdown(f"**3. Durbin-Watson Test (Autocorrelation):** `{dw:.4f}`")
                
                bp_test = het_breuschpagan(model.resid, model.model.exog)
                st.markdown(f"**4. Breusch-Pagan Test (Heteroscedasticity P-value):** `{bp_test[1]:.4f}`")
        else:
            st.warning("ဂဏန်းကော်လံ အလုံအလောက် လိုအပ်ပါသည်။")

    elif module == "4. Comparative Tests (T-Test, ANOVA)":
        st.subheader("⚖️ Comparative Hypothesis Testing")
        test_choice = st.selectbox("Choose Test", ["Independent T-Test (2 Groups)", "One-Way ANOVA (3+ Groups)"])
        
        if test_choice == "Independent T-Test (2 Groups)":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                g_col = st.selectbox("Group Column (Categorical)", cat_cols)
                v_col = st.selectbox("Value Column (Numerical)", num_cols)
                groups = df_clean[g_col].dropna().unique()
                if len(groups) == 2:
                    if st.button("Run T-Test"):
                        sub_df = df_clean[[g_col, v_col]].dropna()
                        d1 = sub_df[sub_df[g_col] == groups[0]][v_col]
                        d2 = sub_df[sub_df[g_col] == groups[1]][v_col]
                        t_stat, p_val = stats.ttest_ind(d1, d2)
                        st.write(f"- T-statistic: `{t_stat:.4f}` | P-value: `{p_val:.4f}`")
                else:
                    st.error("အဆိုပါကော်လံတွင် အုပ်စု ၂ ခုတိတိ ရှိရပါမည်။")

    elif module == "5. AI Executive Summary & Report":
        st.subheader("📝 Automated Thesis Summary Report")
        if st.button("Generate Report"):
            st.markdown("---")
            st.markdown("### 📊 SURVEY ANALYSIS REPORT")
            st.write(f"- Total valid responses analyzed: **{len(df_clean)} rows**")
            st.write(f"- Total variables cleaned and processed successfully.")
            st.success("✅ သုတေသနစာတမ်းအတွက် အချက်အလက်များ အဆင်သင့် ဖြစ်ပါပြီ။")

else:
    st.info("👋 စတင်ရန် ကျေးဇူးပြု၍ သင့်ရဲ့ survey.xlsx ဖိုင်ကို အထက်ပါနေရာတွင် တင်ပေးပါ။")
