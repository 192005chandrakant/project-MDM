import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a random forest model
    """
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine model
    """
    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, f1, report

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type="logistic"):
    """
    Train and evaluate a model based on the specified type
    """
    if model_type == "logistic":
        model = train_logistic_regression(X_train, y_train)
    elif model_type == "rf":
        model = train_random_forest(X_train, y_train)
    elif model_type == "svm":
        model = train_svm(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    acc, f1, report = evaluate_model(model, X_test, y_test)
    return acc, f1, report, model