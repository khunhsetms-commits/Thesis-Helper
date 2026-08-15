import streamlit as st
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thesis Helper AI", layout="wide")
st.title("🚀 Automated Thesis Data Analyzer (End-to-End)")
st.markdown("ဒေတာဖိုင်တင်လိုက်ရုံဖြင့် အလိုအလျောက် သန့်စင်ခြင်း၊ စာရင်းအင်းစစ်ဆေးခြင်းနှင့် ရလဒ်ထုတ်ပေးခြင်းတို့ကို တစ်ပြိုင်နက် လုပ်ဆောင်ပေးမည်။")

uploaded_file = st.file_uploader("📊 Excel သို့မဟုတ် CSV ဖိုင် တင်ပါ", type=["csv", "xlsx"])

if uploaded_file:
    # 1. Load Data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    st.subheader("၁. မူရင်းဒေတာ အနှစ်ချုပ် (Raw Data Preview)")
    st.dataframe(df.head())
    
    # 2. Automated Data Cleaning
    st.subheader("၂. အလိုအလျောက် ဒေတာသန့်စင်ခြင်း (Auto-Cleaning)")
    initial_rows = len(df)
    df = df.dropna()  # Missing values များကို ဖယ်ရှားခြင်း
    cleaned_rows = len(df)
    st.info(f"✨ ဒေတာသန့်စင်ပြီးစီးပါပြီ။ (မူရင်းအတန်းရေ: {initial_rows} | လက်ရှိအတန်းရေ: {cleaned_rows})")
    
    # 3. Automatic Column Classification
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"📈 **Numerical Columns ({len(num_cols)} ခု):**", num_cols)
    with col2:
        st.write(f"📁 **Categorical Columns ({len(cat_cols)} ခု):**", cat_cols)
        
    # 4. Automated Statistical Analysis
    st.subheader("၃. အလိုအလျောက် စာရင်းအင်းခွဲခြမ်းစိတ်ဖြာမှု (Auto-Analysis Results)")
    
    # Descriptive Statistics for all numerical columns
    st.markdown("#### 📊 Descriptive Statistics (Descriptive ဇယား)")
    st.dataframe(df[num_cols].describe())
    
    # Correlation Matrix if more than 1 numerical column
    if len(num_cols) >= 2:
        st.markdown("#### 🔗 Correlation Analysis (ဆက်စပ်မှု ဆန်းစစ်ချက်)")
        corr_matrix = df[num_cols].corr()
        st.dataframe(corr_matrix)
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    
    # Automated T-Test / Group Comparison if both Cat and Num cols exist
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        st.markdown("#### 🧪 Group Comparison (Independent T-Test)")
        group_col = cat_cols[0]
        val_col = num_cols[0]
        
        groups = df[group_col].unique()
        if len(groups) == 2:
            data1 = df[df[group_col] == groups[0]][val_col]
            data2 = df[df[group_col] == groups[1]][val_col]
            
            t_stat, p_val = stats.ttest_ind(data1, data2)
            st.write(f"**Comparing `{val_col}` across groups in `{group_col}`:**")
            st.write(f"- T-statistic: `{t_stat:.4f}`")
            st.write(f"- P-value: `{p_val:.4f}`")
            
            if p_val < 0.05:
                st.success("✅ **Conclusion:** Statistically Significant (အုပ်စုများကြား သိသာထင်ရှားသော ကွာခြားချက် ရှိပါသည်)")
            else:
                st.warning("❌ **Conclusion:** Not Significant (အုပ်စုများကြား သိသာထင်ရှားသော ကွာခြားချက် မရှိပါ)")
        else:
            st.info("ℹ️ T-test လုပ်ဆောင်ရန် Categorical ကော်လံတွင် အုပ်စု ၂ ခုသာ ရှိရန် လိုအပ်ပါသည်။")

    st.success("🎉 အချက်အလက် ခွဲခြမ်းစိတ်ဖြာမှု အားလုံး အောင်မြင်စွာ ပြီးဆုံးပါပြီ။")

else:
    st.info("👋 စတင်ရန် ကျေးဇူးပြု၍ သင့်ရဲ့ Research/Business Dataset ဖိုင်ကို အထက်ပါနေရာတွင် တင်ပေးပါ။")
