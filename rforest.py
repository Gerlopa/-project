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
from sklearn.ensemble import RandomForestClassifier
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


def run_forest():
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

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        oob_score=True
    )
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
    oob       = round(model.oob_score_, 3)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
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
        cmap="Greens"
    )
    ax.set_title("Confusion Matrix (n=1000)", fontsize=13, fontweight="bold")
    confusion_graph = fig_to_base64()

    # =========================
    # 2. FEATURE IMPORTANCE + ERROR BARS
    # =========================
    fig, ax = plt.subplots(figsize=(7, 4))
    importances = np.array([tree.feature_importances_ for tree in model.estimators_])
    mean_imp    = importances.mean(axis=0)
    std_imp     = importances.std(axis=0)
    sorted_idx  = np.argsort(mean_imp)

    ax.barh(
        [features[i] for i in sorted_idx],
        mean_imp[sorted_idx],
        xerr=std_imp[sorted_idx],
        color="#27ae60", ecolor="#2c3e50",
        capsize=5, alpha=0.85
    )
    ax.set_title("Feature Importance ± Std Between Trees", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Importance")
    feature_graph = fig_to_base64()

    # =========================
    # 3. OOB ERROR vs N_ESTIMATORS
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    oob_errors      = []
    estimator_range = range(10, 210, 10)

    for n in estimator_range:
        rf = RandomForestClassifier(
            n_estimators=n,
            random_state=42,
            oob_score=True
        )
        rf.fit(X_train_sc, y_train)
        oob_errors.append(1 - rf.oob_score_)

    ax.plot(estimator_range, oob_errors, color="#27ae60", linewidth=2, marker="o", markersize=4)
    ax.set_title("OOB Error vs Number of Trees", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("OOB Error")
    ax.grid(alpha=0.3)
    ax.axhline(y=min(oob_errors), color="#e74c3c", linestyle="--", alpha=0.7, label=f"Min: {round(min(oob_errors),3)}")
    ax.legend()
    oob_graph = fig_to_base64()

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
    class_labels = le.classes_
    colors_prob  = ["#e74c3c", "#f39c12", "#2ecc71"]

    for i, (cls, color) in enumerate(zip(class_labels, colors_prob)):
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
    bar_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white")
    ax.set_title("Risk Distribution in Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    risk_graph = fig_to_base64()

    max_idx     = np.argmax(mean_imp)
    top_feature = features[max_idx]
    top_value   = round(mean_imp[max_idx], 3)
    insight     = f"The most influential variable is '{top_feature}' with importance {top_value} (±{round(std_imp[max_idx],3)})"

    return {
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "cv":              cv,
        "oob":             oob,
        "confusion_graph": confusion_graph,
        "feature_graph":   feature_graph,
        "oob_graph":       oob_graph,
        "roc_graph":       roc_graph,
        "proba_graph":     proba_graph,
        "risk_graph":      risk_graph,
        "top_feature":     top_feature,
        "top_value":       top_value,
        "insight":         insight
    }


def predict_forest(porcentaje_led, total):
    df, features = load_data()
    X = df[features]
    y = df["risk"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    pipeline.fit(X, y_enc)
    sample = np.array([[porcentaje_led, total]])
    pred = pipeline.predict(sample)
    return le.inverse_transform(pred)[0]
