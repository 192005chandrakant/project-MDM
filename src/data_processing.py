import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression

def preprocess_data(df, target_col):
    """
    Preprocess the data by scaling features and separating target
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle non-numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found in the dataset")
    
    X_numeric = X[numeric_columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    return X_scaled, y, X_numeric.columns


def apply_pca(X_train, X_test, n_components=5):
    """
    Apply PCA dimensionality reduction
    """
    n_components = min(n_components, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    explained_var = pca.explained_variance_ratio_
    return X_train_pca, X_test_pca, explained_var


def apply_chi2(X_train, y_train, X_test, k=5):
    """
    Apply Chi2 feature selection
    """
    # Ensure data is positive for chi2
    X_train_abs = np.abs(X_train)
    X_test_abs = np.abs(X_test)
    
    # Add small constant to avoid zero values
    X_train_abs = X_train_abs + 1e-10
    X_test_abs = X_test_abs + 1e-10
    
    k = min(k, X_train.shape[1])
    selector = SelectKBest(chi2, k=k)
    X_train_new = selector.fit_transform(X_train_abs, y_train)
    X_test_new = selector.transform(X_test_abs)
    selected_features = selector.get_support(indices=True)
    return X_train_new, X_test_new, selected_features


def apply_rfe(X_train, y_train, X_test, k=5):
    """
    Apply Recursive Feature Elimination
    """
    k = min(k, X_train.shape[1])
    model = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(model, n_features_to_select=k)
    X_train_new = rfe.fit_transform(X_train, y_train)
    X_test_new = rfe.transform(X_test)
    selected_features = rfe.get_support(indices=True)
    return X_train_new, X_test_new, selected_features


def apply_mutual_info(X_train, y_train, X_test, k=5):
    """
    Apply Mutual Information feature selection
    """
    k = min(k, X_train.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)
    selected_features = selector.get_support(indices=True)
    return X_train_new, X_test_new, selected_features