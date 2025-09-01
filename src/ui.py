import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import preprocess_data, apply_pca, apply_chi2, apply_rfe, apply_mutual_info

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import preprocess_data, apply_pca, apply_chi2, apply_rfe, apply_mutual_info
from src.model import train_and_evaluate

st.set_page_config(page_title="Data Reduction Tool", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("� Data Reduction Tool")
    st.markdown("""
    <span style='color:#3498db;font-size:1.1em;'>Professional Feature Selection & Dimensionality Reduction</span>
    <hr style='border:1px solid #3498db;'>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown("""
    <small>Supported: CSV files. For Excel, use the main app.</small>
    """, unsafe_allow_html=True)

# --- Main Content ---
st.markdown("""
<style>
.section-header {
    background: linear-gradient(90deg, #e3f2fd 0%, #f8f9fa 100%);
    padding: 0.7rem 1.2rem;
    border-radius: 8px;
    border-left: 4px solid #3498db;
    margin: 1.2rem 0 0.7rem 0;
    font-weight: 600;
    color: #2c3e50;
    font-size: 1.1rem;
}
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(44,62,80,0.07);
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid #e3e3e3;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

if 'uploaded_file' not in locals():
    uploaded_file = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-header">🎯 Target Variable Selection</div>', unsafe_allow_html=True)
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        X_scaled, y, feature_names = preprocess_data(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        st.markdown('<div class="section-header">⚙️ Feature Selection Method</div>', unsafe_allow_html=True)
        method = st.radio(
            "Choose Dimensionality Reduction Method",
            ("PCA", "Chi2 Feature Selection", "RFE Feature Selection", "Mutual Information Feature Selection"),
            horizontal=True
        )

        n_features = st.slider(
            "Select Number of Components/Features", 2, min(10, X_train.shape[1]), 5
        )

        st.markdown('<div class="section-header">🤖 Model Selection</div>', unsafe_allow_html=True)
        model_type = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Support Vector Machine"])
        if model_type == "Logistic Regression":
            model_key = "logistic"
        elif model_type == "Random Forest":
            model_key = "rf"
        else:
            model_key = "svm"

        if st.button("Run Reduction & Train Model", use_container_width=True):
            if method == "PCA":
                X_train_new, X_test_new, explained_var = apply_pca(
                    X_train, X_test, n_components=n_features
                )
                st.success("✅ PCA Applied")
                st.write("Explained Variance Ratio:", explained_var)

            elif method == "Chi2 Feature Selection":
                X_train_new, X_test_new, selected = apply_chi2(
                    X_train, y_train, X_test, k=n_features
                )
                st.success("✅ Chi2 Feature Selection Applied")
                st.write("Selected Features:", feature_names[selected].tolist())

            elif method == "RFE Feature Selection":
                X_train_new, X_test_new, selected = apply_rfe(
                    X_train, y_train, X_test, k=n_features
                )
                st.success("✅ RFE Feature Selection Applied")
                st.write("Selected Features:", feature_names[selected].tolist())

            elif method == "Mutual Information Feature Selection":
                X_train_new, X_test_new, selected = apply_mutual_info(
                    X_train, y_train, X_test, k=n_features
                )
                st.success("✅ Mutual Information Feature Selection Applied")
                st.write("Selected Features:", feature_names[selected].tolist())

            acc, f1, report, model = train_and_evaluate(
                X_train_new, y_train, X_test_new, y_test, model_type=model_key
            )

            st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <b>Accuracy</b><br><span style='font-size:1.5em;color:#27ae60'>{acc:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <b>F1 Score</b><br><span style='font-size:1.5em;color:#2980b9'>{f1:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            reduced_df = pd.DataFrame(X_train_new)
            reduced_df["target"] = y_train.reset_index(drop=True)
            csv = reduced_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Reduced Dataset (Train)", csv, "reduced_train.csv", "text/csv")

else:
    st.info("👈 Upload a CSV file to get started! Use the sidebar.")