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
import re

# Page Configuration
st.set_page_config(
    page_title="Survey Thesis Analyzer Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("### 🎓 Survey Dataset AI Thesis & Statistical Analysis Pro")
st.markdown("မြန်မာနိုင်ငံဆိုင်ရာ Survey ဒေတာများအတွက် Reliability Analysis အပါအဝင် အလိုအလျောက် စာရင်းအင်းစစ်ဆေးခြင်း စနစ်။")
st.markdown("---")

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

# Cronbach's Alpha Calculation Function
def calculate_cronbach_alpha(df_subset):
    item_vars = df_subset.var(axis=0, ddof=1)
    total_score = df_subset.sum(axis=1)
    total_var = total_score.var(ddof=1)
    n_items = df_subset.shape[1]
    
    if n_items < 2 or total_var == 0:
        return np.nan
    
    alpha = (n_items / (n_items - 1)) * (1 - (item_vars.sum() / total_var))
    return alpha

# Cached Data Loader & Cleaner
@st.cache_data
def load_and_clean_data(uploaded_file, file_extension):
    if file_extension == 'xlsx':
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    df_clean = df.copy()
    
    if len(df_clean.columns) > 2:
        age_col = df_clean.columns[2]
        df_clean[age_col] = df_clean[age_col].apply(extract_number)
    
    for col_idx in [14, 15]:
        if col_idx < len(df_clean.columns):
            c_name = df_clean.columns[col_idx]
            df_clean[c_name] = df_clean[c_name].apply(extract_number)

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            converted = df_clean[col].apply(extract_number)
            if converted.notnull().mean() > 0.5:
                df_clean[col] = converted

    if len(df_clean.columns) > 2:
        df_clean = df_clean.dropna(subset=[df_clean.columns[2]])
        
    return df, df_clean

# File Uploader
uploaded_file = st.file_uploader("📊 Survey Excel (.xlsx) သို့မဟုတ် CSV ဖိုင် တင်ပါ", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = 'xlsx' if uploaded_file.name.endswith('.xlsx') else 'csv'
    raw_df, df_clean = load_and_clean_data(uploaded_file, file_ext)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("မူရင်း အတန်းရေ (Rows)", len(raw_df))
    with col2:
        st.metric("သန့်စင်ပြီး အတန်းရေ", len(df_clean))
    with col3:
        st.metric("ကော်လံရေ (Columns)", len(df_clean.columns))
        
    st.success("✨ ဒေတာ တင်သွင်းခြင်းနှင့် သန့်စင်ခြင်း အောင်မြင်ပါသည်။")
    
    num_cols = df_clean.select_dtypes(include='number').columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Sidebar Navigation
    st.sidebar.header("⚙️ Analysis Modules")
    module = st.sidebar.selectbox("Choose Module", [
        "1. Cleaned Data & Descriptive Stats",
        "2. Reliability Analysis (Cronbach's Alpha)",
        "3. Regression Models (Simple, Multiple, Logistic)",
        "4. Goodness of Fit & Diagnostics",
        "5. Comparative Tests (T-Test, ANOVA)",
        "6. AI Executive Summary & Report"
    ])
    
    # MODULE 1: Descriptive Stats
    if module == "1. Cleaned Data & Descriptive Stats":
        st.subheader("📋 Cleaned Dataset Preview & Summary")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        csv_data = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_survey_data.csv",
            mime="text/csv"
        )
        
        st.markdown("#### Numerical Variables Descriptive Statistics")
        st.dataframe(df_clean[num_cols].describe(), use_container_width=True)
        
        if len(num_cols) >= 2:
            st.markdown("#### Correlation Matrix & Heatmap")
            corr = df_clean[num_cols].corr()
            st.dataframe(corr, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
            st.pyplot(fig)

    # MODULE 2: Reliability Analysis (Cronbach's Alpha)
    elif module == "2. Reliability Analysis (Cronbach's Alpha)":
        st.subheader("📊 Reliability Analysis (Cronbach's Alpha)")
        st.markdown(" Likert Scale မေးခွန်းစုများ (ဥပမာ- သဘောထားတိုင်းတာချက်များ) ၏ အတွင်းပိုင်း တည်ငြိမ်ယုံကြည်စိတ်ချရမှု (Internal Consistency) ကို စစ်ဆေးရန် ကော်လံများကို ရွေးချယ်ပါ။")
        
        if len(num_cols) >= 2:
            selected_items = st.multiselect("Scale Items ရွေးချယ်ပါ (Variables အနည်းဆုံး ၂ ခု)", num_cols)
            
            if selected_items:
                if st.button("Calculate Cronbach's Alpha"):
                    sub_df = df_clean[selected_items].dropna()
                    alpha = calculate_cronbach_alpha(sub_df)
                    
                    st.markdown("---")
                    st.metric("Cronbach's Alpha ($\alpha$)", f"{alpha:.4f}" if not np.isnan(alpha) else "N/A")
                    
                    # Interpretation Guide
                    if not np.isnan(alpha):
                        if alpha >= 0.9:
                            st.success("🟢 အလွန်ကောင်းမွန်သော ယုံကြည်စိတ်ချရမှု (Excellent)")
                        elif alpha >= 0.8:
                            st.success("🟢 ကောင်းမွန်သော ယုံကြည်စိတ်ချရမှု (Good)")
                        elif alpha >= 0.7:
                            st.warning("🟡 လက်ခံနိုင်လောက်သော ယုံကြည်စိတ်ချရမှု (Acceptable)")
                        elif alpha >= 0.6:
                            st.warning("🟠 သတိပြုရမည့် အခြေအနေ (Questionable)")
                        else:
                            st.error("🔴 ယုံကြည်စိတ်ချရမှု မရှိပါ (Poor)")
                            
                    st.markdown("#### Item-wise Statistics")
                    item_summary = pd.DataFrame({
                        "Mean": sub_df.mean(),
                        "Std Deviation": sub_df.std(),
                        "Item-Total Correlation": [sub_df[col].corr(sub_df.sum(axis=1) - sub_df[col]) for col in selected_items]
                    })
                    st.dataframe(item_summary, use_container_width=True)
        else:
            st.warning("Reliability စစ်ဆေးရန် ဂဏန်းကော်လံ အလုံအလောက် မရှိပါ။")

    # MODULE 3: Regression Models
    elif module == "3. Regression Models (Simple, Multiple, Logistic)":
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

    # MODULE 4: Goodness of Fit & Diagnostics
    elif module == "4. Goodness of Fit & Diagnostics":
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

    # MODULE 5: Comparative Tests (T-Test, ANOVA)
    elif module == "5. Comparative Tests (T-Test, ANOVA)":
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
                    
        elif test_choice == "One-Way ANOVA (3+ Groups)":
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                g_col = st.selectbox("Group Column (Categorical 3+)", cat_cols)
                v_col = st.selectbox("Value Column (Numerical)", num_cols)
                groups = df_clean[g_col].dropna().unique()
                if len(groups) >= 3:
                    if st.button("Run ANOVA"):
                        sub_df = df_clean[[g_col, v_col]].dropna()
                        group_data = [sub_df[sub_df[g_col] == g][v_col] for g in groups]
                        f_stat, p_val = stats.f_oneway(*group_data)
                        st.write(f"- F-statistic: `{f_stat:.4f}` | P-value: `{p_val:.4f}`")
                else:
                    st.warning("ANOVA အတွက် အုပ်စု အနည်းဆုံး ၃ ခု လိုအပ်ပါသည်။")

    # MODULE 6: Executive Summary Report
    elif module == "6. AI Executive Summary & Report":
        st.subheader("📝 Automated Thesis Summary Report")
        if st.button("Generate Final Report"):
            st.markdown("---")
            st.markdown("### 📊 SURVEY ANALYSIS REPORT")
            st.write(f"- Total raw responses: **{len(raw_df)} rows**")
            st.write(f"- Total valid responses analyzed after cleaning: **{len(df_clean)} rows**")
            st.success("✅ သုတေသနစာတမ်းအတွက် အချက်အလက်များ အဆင်သင့် ဖြစ်ပါပြီ။")

else:
    st.info("👋 စတင်ရန် ကျေးဇူးပြု၍ သင့်ရဲ့ survey ဖိုင် (.xlsx သို့မဟုတ် .csv) ကို အထက်ပါနေရာတွင် တင်ပေးပါ။")
                  
