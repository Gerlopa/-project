import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, auc
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "datos luz.csv")


def load_data():
    df = pd.read_csv(file_path, sep=";", encoding="latin1")
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    df.rename(columns={"%ledxloca": "porcentaje_led"}, inplace=True)
    df["porcentaje_led"] = df["porcentaje_led"].astype(str)
    df["porcentaje_led"] = df["porcentaje_led"].str.replace("%", "")
    df["porcentaje_led"] = df["porcentaje_led"].str.replace(",", ".")
    df["porcentaje_led"] = pd.to_numeric(df["porcentaje_led"], errors="coerce")
    df["total"] = df["total"].replace(0, 1)
    ratio = (df["mh"] + df["na"]) / df["total"]

    def classify_risk(x):
        if x > 0.5:
            return "High"
        elif x > 0.2:
            return "Medium"
        else:
            return "Low"

    df["risk"] = ratio.apply(classify_risk)
    features = ["porcentaje_led", "total"]
    df = df.dropna(subset=features + ["risk"])
    return df, features


def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return encoded


def run_logistic():
    df, features = load_data()

    X = df[features]
    y = df["risk"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)

    X_all_sc   = scaler.transform(X)
    y_all_pred = model.predict(X_all_sc)

    # =========================
    # METRICS
    # =========================
    accuracy  = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3)
    recall    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3)
    f1        = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    cv = round(cross_val_score(pipeline, X, y_enc, cv=5, scoring="accuracy").mean(), 3)

    # =========================
    # 1. CONFUSION MATRIX
    # =========================
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_enc, y_all_pred,
        display_labels=le.classes_,
        ax=ax, colorbar=False,
        cmap="Blues"
    )
    ax.set_title("Confusion Matrix (n=1000)", fontsize=13, fontweight="bold")
    confusion_graph = fig_to_base64()

    # =========================
    # 2. COEFFICIENTS BY CLASS
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5))
    coef = model.coef_
    x = np.arange(len(features))
    width = 0.25
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    for i, (cls, color) in enumerate(zip(le.classes_, colors)):
        ax.bar(x + i * width, coef[i], width, label=cls, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(features)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Coefficients by Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("Coefficient (+ increases risk, - reduces it)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    coef_graph = fig_to_base64()

    # =========================
    # 3. SIGMOID CURVE
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    x_values   = np.linspace(df["porcentaje_led"].min(), df["porcentaje_led"].max(), 200)
    total_mean = df["total"].mean()
    X_plot     = np.array([[x, total_mean] for x in x_values])
    X_plot_sc  = scaler.transform(X_plot)
    probs      = model.predict_proba(X_plot_sc)

    for i, (cls, color) in enumerate(zip(le.classes_, ["#e74c3c", "#f39c12", "#2ecc71"])):
        ax.plot(x_values, probs[:, i], label=cls, color=color, linewidth=2)

    ax.set_title("Sigmoid Curve by Class vs % LED", fontsize=13, fontweight="bold")
    ax.set_xlabel("% LED")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(alpha=0.3)
    logistic_graph = fig_to_base64()

    # =========================
    # 4. ROC CURVE ✅ NUEVO
    # =========================
    fig, ax = plt.subplots(figsize=(8, 6))
    classes    = le.classes_
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    colors_roc = ["#e74c3c", "#f39c12", "#2ecc71"]

    for i, (cls, color) in enumerate(zip(classes, colors_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc     = round(auc(fpr, tpr), 3)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls} (AUC = {roc_auc})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")
    ax.set_title("ROC Curve — One vs Rest", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    roc_graph = fig_to_base64()

    # =========================
    # 5. PREDICTION PROBABILITIES
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (cls, color) in enumerate(zip(le.classes_, ["#e74c3c", "#f39c12", "#2ecc71"])):
        ax.hist(
            y_proba[:, i], bins=15, alpha=0.6,
            label=cls, color=color, edgecolor="white"
        )
    ax.set_title("Prediction Probability Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    proba_graph = fig_to_base64()

    # =========================
    # 6. RISK DISTRIBUTION
    # =========================
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["risk"].value_counts().reindex(["High", "Medium", "Low"])
    counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="white")
    ax.set_title("Risk Distribution in Dataset", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    risk_graph = fig_to_base64()

    abs_coef    = np.abs(coef).mean(axis=0)
    max_idx     = np.argmax(abs_coef)
    top_feature = features[max_idx]
    top_value   = round(abs_coef[max_idx], 3)
    insight     = f"The most influential variable is '{top_feature}' with mean absolute coefficient of {top_value}"

    return {
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "cv":              cv,
        "confusion_graph": confusion_graph,
        "coef_graph":      coef_graph,
        "logistic_graph":  logistic_graph,
        "roc_graph":       roc_graph,
        "proba_graph":     proba_graph,
        "risk_graph":      risk_graph,
        "top_feature":     top_feature,
        "top_value":       top_value,
        "insight":         insight
    }


def predict_risk(porcentaje_led, total):
    df, features = load_data()
    X = df[features]
    y = df["risk"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X, y_enc)
    sample = np.array([[porcentaje_led, total]])
    pred   = pipeline.predict(sample)
    return le.inverse_transform(pred)[0]
