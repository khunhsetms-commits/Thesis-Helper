import streamlit as st
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Thesis Helper", layout="wide")
st.title("🤖 AI-Powered Thesis Helper & Auto-Analyzer")
st.markdown("ဒေတာဖိုင်တင်လိုက်တာနဲ့ AI က သင့်ဒေတာအတွက် အသင့်တော်ဆုံး Statistical Test ကို ရွေးချယ်ပေးပြီး အလိုအလျောက် တွက်ချက်ပေးပါမည်။")

uploaded_file = st.file_uploader("Excel သို့မဟုတ် CSV ဖိုင် တင်ပါ", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())
    
    # Column selections
    columns = df.columns.tolist()
    st.sidebar.header("🔍 Test Configuration")
    test_mode = st.sidebar.radio("Analysis Mode", ["Auto-Detect (AI Recommendation)", "Manual Selection"])
    
    if test_mode == "Auto-Detect (AI Recommendation)":
        st.subheader("🤖 AI Recommendation & Auto-Run")
        
        # Simple Logic to recommend tests based on data types
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.write(f"- **Numerical Columns (ဂဏန်းဒေတာ):** {len(num_cols)} ခုတွေ့ရှိသည်")
        st.write(f"- **Categorical Columns (အုပ်စုခွဲဒေတာ):** {len(cat_cols)} ခုတွေ့ရှိသည်")
        
        if len(num_cols) >= 2:
            st.info("💡 **Recommendation:** Numerical ကော်လံ ၂ ခုပါရှိသောကြောင့် **Correlation Analysis** ပြုလုပ်ရန် သင့်လျော်ပါသည်။")
            if st.button("Run Auto-Correlation"):
                corr_matrix = df[num_cols].corr()
                st.write(corr_matrix)
                fig, ax = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap="Blues", ax=ax)
                st.pyplot(fig)
                
        elif len(cat_cols) >= 1 and len(num_cols) >= 1:
            st.info("💡 **Recommendation:** Group ခွဲရန် Categorical ကော်လံနှင့် တိုင်းတာရန် Numerical ကော်လံပါရှိသောကြောင့် **Independent T-Test (သို့) ANOVA** ပြုလုပ်ရန် သင့်လျော်ပါသည်။")
            
    else:
        # Manual Selection Mode (SPSS style)
        st.subheader("⚙️ Manual Statistical Tests")
        selected_test = st.selectbox("Choose Test", ["Descriptive Statistics", "Correlation", "T-Test"])
        
        if selected_test == "Descriptive Statistics":
            st.write(df.describe())
            
        elif selected_test == "Correlation":
            c1 = st.selectbox("Variable 1", df.select_dtypes(include='number').columns)
            c2 = st.selectbox("Variable 2", df.select_dtypes(include='number').columns)
            if st.button("Calculate Correlation"):
                corr, p_val = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
                st.success(f"Pearson Correlation (r): {corr:.4f}")
                st.success(f"P-value: {p_val:.4f}")
                if p_val < 0.05:
                    st.markdown("✅ **Result:** Statistically Significant (ဆက်စပ်မှုရှိပါသည်)")
                else:
                    st.markdown("❌ **Result:** Not Significant (ဆက်စပ်မှုမရှိပါ)")

else:
    st.info("စတင်ရန် ကျေးဇူးပြု၍ ဒေတာဖိုင် တင်ပေးပါ။")
