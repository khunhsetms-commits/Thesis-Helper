import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ultimate Thesis Analyzer Pro", layout="wide")

st.title("🎓 Ultimate AI Thesis & Statistical Analysis Pro")
st.markdown("SPSS ကဲ့သို့ အဆင့်မြင့် Test များ (Regression အမျိုးမျိုး၊ Goodness of Fit၊ T-Test၊ ANOVA) ကို တစ်နေရာတည်းတွင် အစအဆုံး အလိုအလျောက် ဆန်းစစ်ပေးသော စနစ်။")

# Burmese to English digit converter
def burmese_to_english_digits(text):
    if isinstance(text, str):
        burmese_digits = "၀၁၂၃၄၅၆၇၈၉"
        english_digits = "0123456789"
        trans_table = str.maketrans(burmese_digits, english_digits)
        return text.translate(trans_table)
    return text

uploaded_file = st.file_uploader("📊 Excel (.xlsx) သို့မဟုတ် CSV ဖိုင် တင်ပါ", type=["csv", "xlsx"])

if uploaded_file:
    # 1. Load Data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    # 2. Data Cleaning & Digit Conversion
    for col in df.columns:
        df[col] = df[col].apply(burmese_to_english_digits)
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
            
    df = df.dropna()
    
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Sidebar Navigation for Features
    st.sidebar.header("⚙️ Analysis Modules")
    module = st.sidebar.selectbox("Choose Module", [
        "1. Overview & Data Summary",
        "2. Regression Models (Simple, Multiple, Logistic)",
        "3. Goodness of Fit & Diagnostics",
        "4. Comparative Tests (T-Test, ANOVA)",
        "5. AI Executive Summary & Report"
    ])
    
    if module == "1. Overview & Data Summary":
        st.subheader("📋 Cleaned Dataset & Descriptive Statistics")
        st.dataframe(df.head())
        st.write(f"Total Rows (Cleaned): {len(df)} | Columns: {len(df.columns)}")
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe())
        
        if len(num_cols) >= 2:
            st.markdown("#### Correlation Matrix")
            corr = df[num_cols].corr()
            st.dataframe(corr)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

    elif module == "2. Regression Models (Simple, Multiple, Logistic)":
        st.subheader("📈 Advanced Regression Analysis Suite")
        reg_type = st.selectbox("Select Regression Type", ["Simple Linear Regression", "Multiple Linear Regression", "Logistic Regression (Binary)"])
        
        if reg_type == "Simple Linear Regression":
            if len(num_cols) >= 2:
                y_var = st.selectbox("Dependent Variable (Y)", num_cols)
                x_var = st.selectbox("Independent Variable (X)", [c for c in num_cols if c != y_var])
                if st.button("Run Simple Regression"):
                    model = smf.ols(f"{y_var} ~ {x_var}", data=df).fit()
                    st.text(str(model.summary()))
            else:
                st.warning("ဂဏန်းကော်လံ ၂ ခု လိုအပ်ပါသည်။")
                
        elif reg_type == "Multiple Linear Regression":
            if len(num_cols) >= 3:
                y_var = st.selectbox("Dependent Variable (Y)", num_cols)
                x_vars = st.multiselect("Independent Variables (X's)", [c for c in num_cols if c != y_var])
                if x_vars and st.button("Run Multiple Regression"):
                    formula = f"{y_var} ~ " + " + ".join(x_vars)
                    model = smf.ols(formula, data=df).fit()
                    st.text(str(model.summary()))
            else:
                st.warning("Multiple Regression အတွက် ဂဏန်းကော်လံ အနည်းဆုံး ၃ ခု လိုအပ်ပါသည်။")
                
        elif reg_type == "Logistic Regression (Binary)":
            if len(num_cols) >= 1 and len(cat_cols) >= 1:
                y_var = st.selectbox("Binary Dependent Variable (Target Group)", cat_cols)
                x_vars = st.multiselect("Predictors (X's)", num_cols)
                if x_vars and st.button("Run Logistic Regression"):
                    # Encode target to 0 and 1
                    df_log = df.copy()
                    target_vals = df_log[y_var].unique()
                    if len(target_vals) == 2:
                        df_log['target_encoded'] = (df_log[y_var] == target_vals[0]).astype(int)
                        formula = f"target_encoded ~ " + " + ".join(x_vars)
                        logit_model = smf.logit(formula, data=df_log).fit()
                        st.text(str(logit_model.summary()))
                    else:
                        st.error("Target variable တွင် အုပ်စု ၂ ခုသာ ရှိရပါမည်။")
            else:
                st.warning("Logistic Regression အတွက် Categorical Dependent တစ်ခုနှင့် Numerical Predictors များ လိုအပ်ပါသည်။")

    elif module == "3. Goodness of Fit & Diagnostics":
        st.subheader("🔍 Model Diagnostics & Goodness of Fit Tests")
        st.markdown("စံသတ်မှတ်ထားသော Regression မော်ဒယ်တစ်ခုဆောက်ပြီး Goodness-of-Fit များကို စစ်ဆေးခြင်း:")
        
        if len(num_cols) >= 2:
            y_var = st.selectbox("Dependent Variable (Y)", num_cols, key="gof_y")
            x_vars = st.multiselect("Independent Variables (X's)", [c for c in num_cols if c != y_var], key="gof_x")
            
            if x_vars and st.button("Run Diagnostic Tests"):
                formula = f"{y_var} ~ " + " + ".join(x_vars)
                model = smf.ols(formula, data=df).fit()
                
                st.markdown(f"**1. R-squared / Adjusted R-squared (Goodness of Fit):** `{model.rsquared:.4f}` / `{model.rsquared_adj:.4f}`")
                st.markdown(f"**2. F-statistic & P-value:** F = `{model.fvalue:.4f}`, P = `{model.f_pvalue:.4e}`")
                
                # Durbin-Watson for Autocorrelation
                dw = durbin_watson(model.resid)
                st.markdown(f"**3. Durbin-Watson Test (Autocorrelation):** `{dw:.4f}` (Ideal: ~2.0)")
                
                # Breusch-Pagan Test for Heteroscedasticity
                bp_test = het_breuschpagan(model.resid, model.model.exog)
                st.markdown(f"**4. Breusch-Pagan Test (Heteroscedasticity P-value):** `{bp_test[1]:.4f}`")
                if bp_test[1] > 0.05:
                    st.success("✅ Homoscedasticity verified (Variance is stable).")
                else:
                    st.warning("⚠️ Heteroscedasticity detected (Variance is not constant).")
        else:
            st.warning("ဂဏန်းကော်လံ အလုံအလောက် လိုအပ်ပါသည်။")

    elif module == "4. Comparative Tests (T-Test, ANOVA)":
        st.subheader("⚖️ Comparative Hypothesis Testing")
        test_choice = st.selectbox("Choose Test", ["Independent T-Test (2 Groups)", "One-Way ANOVA (3+ Groups)"])
        
        if test_choice == "Independent T-Test (2 Groups)":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                g_col = st.selectbox("Group Column (Categorical)", cat_cols)
                v_col = st.selectbox("Value Column (Numerical)", num_cols)
                groups = df[g_col].unique()
                if len(groups) == 2:
                    if st.button("Run T-Test"):
                        d1 = df[df[g_col] == groups[0]][v_col]
                        d2 = df[df[g_col] == groups[1]][v_col]
                        t_stat, p_val = stats.ttest_ind(d1, d2)
                        st.write(f"- T-statistic: `{t_stat:.4f}` | P-value: `{p_val:.4f}`")
                        if p_val < 0.05:
                            st.success("✅ Statistically Significant difference found.")
                        else:
                            st.warning("❌ No significant difference.")
                else:
                    st.error("အဆိုပါကော်လံတွင် အုပ်စု ၂ ခုတိတိ ရှိရပါမည်။")
                    
        elif test_choice == "One-Way ANOVA (3+ Groups)":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                g_col = st.selectbox("Group Column", cat_cols, key="anova_g")
                v_col = st.selectbox("Value Column", num_cols, key="anova_v")
                if st.button("Run ANOVA"):
                    groups_data = [group[v_col].values for name, group in df.groupby(g_col)]
                    if len(groups_data) >= 3:
                        f_stat, p_val = stats.f_oneway(*groups_data)
                        st.write(f"- F-statistic: `{f_stat:.4f}` | P-value: `{p_val:.4f}`")
                        if p_val < 0.05:
                            st.success("✅ Significant difference across groups.")
                        else:
                            st.warning("❌ No significant difference across groups.")
                    else:
                        st.info("ANOVA အတွက် အုပ်စု အနည်းဆုံး ၃ ခု ပါဝင်ရပါမည်။")

    elif module == "5. AI Executive Summary & Report":
        st.subheader("📝 Automated Thesis Summary & Interpretation Report")
        st.markdown("တင်သွင်းထားသော ဒေတာနှင့် ဆန်းစစ်ချက်များအပေါ် မူတည်၍ တရားဝင် သုတေသနအစီရင်ခံစာ အနှစ်ချုပ်ကို ထုတ်ပေးခြင်း:")
        
        if st.button("Generate Summary Report"):
            st.markdown("---")
            st.markdown("### 📊 EXECUTIVE SUMMARY & FINDINGS")
            st.write(f"1. **Dataset Overview:** Total valid observations analyzed = **{len(df)} rows** after automated cleaning and missing value removal.")
            st.write(f"2. **Variables Profile:** Identified **{len(num_cols)} numerical variables** and **{len(cat_cols)} categorical variables**.")
            st.write("3. **Statistical Inference:** All parameters were computed using standard Python statistical libraries (SciPy & Statsmodels) matching SPSS analytical standards.")
            st.markdown("✅ **Conclusion:** This automated report can be directly integrated into your thesis research methodology and results chapters.")
            st.markdown("---")

else:
    st.info("👋 စတင်ရန် ကျေးဇူးပြု၍ သင့်ရဲ့ Research Dataset ဖိုင်ကို အထက်ပါနေရာတွင် တင်ပေးပါ။")
