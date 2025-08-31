import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import preprocess_data, apply_pca, apply_chi2, apply_rfe
from src.model import train_and_evaluate

st.set_page_config(page_title="Data Reduction Tool", layout="wide")
st.title("📉 Data Reduction using PCA & Feature Selection")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        X_scaled, y, feature_names = preprocess_data(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        method = st.radio(
            "Choose Dimensionality Reduction Method",
            ("PCA", "Chi2 Feature Selection", "RFE Feature Selection"),
        )

        n_features = st.slider(
            "Select Number of Components/Features", 2, min(10, X_train.shape[1]), 5
        )

        if st.button("Run Reduction & Train Model"):
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

            else:
                X_train_new, X_test_new, selected = apply_rfe(
                    X_train, y_train, X_test, k=n_features
                )
                st.success("✅ RFE Feature Selection Applied")
                st.write("Selected Features:", feature_names[selected].tolist())

            model_type = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])
            model_key = "logistic" if model_type == "Logistic Regression" else "rf"

            acc, f1, report, model = train_and_evaluate(
                X_train_new, y_train, X_test_new, y_test, model_type=model_key
            )

            st.subheader("📊 Model Performance")
            st.write(f"**Accuracy:** {acc:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")

            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            reduced_df = pd.DataFrame(X_train_new)
            reduced_df["target"] = y_train.reset_index(drop=True)
            csv = reduced_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Reduced Dataset (Train)", csv, "reduced_train.csv", "text/csv")