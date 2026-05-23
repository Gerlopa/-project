import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    ConfusionMatrixDisplay
)

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "datos luz.csv")


# =========================
# PREPARAR DATOS
# =========================
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

    #  Sin mh ni na
    features = ["porcentaje_led", "total"]
    df = df.dropna(subset=features + ["risk"])

    return df, features


# =========================
# HELPER
# =========================
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return encoded


# =========================
# MAIN FUNCTION
# =========================
def run_logistic():

    df, features = load_data()

    X = df[features]
    y = df["risk"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ✅ Split antes de escalar
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

    # =========================
    # METRICS
    # =========================
    accuracy  = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3)
    recall    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3)
    f1        = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3)

    # Pipeline para CV sin leakage
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
        y_test, y_pred,
        display_labels=le.classes_,
        ax=ax, colorbar=False,
        cmap="Blues"
    )
    ax.set_title("Matriz de Confusión", fontsize=13, fontweight="bold")
    confusion_graph = fig_to_base64()

    # =========================
    # 2. COEFICIENTES (feature importance de LR)
    #    — muestra signo: positivo aumenta riesgo, negativo lo reduce
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5))
    coef = model.coef_  # shape: (n_classes, n_features)
    x = np.arange(len(features))
    width = 0.25
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    for i, (cls, color) in enumerate(zip(le.classes_, colors)):
        ax.bar(x + i * width, coef[i], width, label=cls, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(features)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Coeficientes por Clase", fontsize=13, fontweight="bold")
    ax.set_ylabel("Coeficiente (+ aumenta riesgo, - lo reduce)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    coef_graph = fig_to_base64()

    # =========================
    # 3. CURVA SIGMOIDE
    #    variando porcentaje_led con total en su media
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))

    x_values   = np.linspace(df["porcentaje_led"].min(), df["porcentaje_led"].max(), 200)
    total_mean = df["total"].mean()

    X_plot = np.array([[x, total_mean] for x in x_values])
    X_plot_sc = scaler.transform(X_plot)
    probs  = model.predict_proba(X_plot_sc)

    colors_sig = ["#e74c3c", "#f39c12", "#2ecc71"]
    for i, (cls, color) in enumerate(zip(le.classes_, colors_sig)):
        ax.plot(x_values, probs[:, i], label=cls, color=color, linewidth=2)

    ax.set_title("Curva Sigmoide por Clase vs % LED", fontsize=13, fontweight="bold")
    ax.set_xlabel("% LED")
    ax.set_ylabel("Probabilidad")
    ax.legend()
    ax.grid(alpha=0.3)
    logistic_graph = fig_to_base64()

    # =========================
    # 4. PREDICTION PROBABILITIES
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (cls, color) in enumerate(zip(le.classes_, ["#e74c3c", "#f39c12", "#2ecc71"])):
        ax.hist(
            y_proba[:, i], bins=15, alpha=0.6,
            label=cls, color=color, edgecolor="white"
        )
    ax.set_title("Distribución de Probabilidades por Clase", fontsize=13, fontweight="bold")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(alpha=0.3)
    proba_graph = fig_to_base64()

    # =========================
    # 5. DISTRIBUCIÓN DE RIESGO
    # =========================
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["risk"].value_counts().reindex(["High", "Medium", "Low"])
    counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="white")
    ax.set_title("Distribución de Riesgo en el Dataset", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cantidad")
    plt.xticks(rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    risk_graph = fig_to_base64()

    # insight
    abs_coef    = np.abs(coef).mean(axis=0)
    max_idx     = np.argmax(abs_coef)
    top_feature = features[max_idx]
    top_value   = round(abs_coef[max_idx], 3)
    insight     = f"La variable más influyente es '{top_feature}' con coeficiente medio absoluto de {top_value}"

    return {
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "cv":              cv,
        "confusion_graph": confusion_graph,
        "coef_graph":      coef_graph,
        "logistic_graph":  logistic_graph,
        "proba_graph":     proba_graph,
        "risk_graph":      risk_graph,
        "top_feature":     top_feature,
        "top_value":       top_value,
        "insight":         insight
    }


# =========================
# PREDICTION
# =========================
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
