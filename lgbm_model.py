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
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    ConfusionMatrixDisplay
)
import lightgbm as lgb

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "datos luz.csv")


# =========================
# DATA PREPARATION
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

    mes_order = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    df["mes_num"] = df["mes"].str.lower().map(mes_order)

    le_loc = LabelEncoder()
    df["localidad_enc"] = le_loc.fit_transform(df["localidad"].astype(str))

    ratio = (df["mh"] + df["na"]) / df["total"]

    def classify_risk(x):
        if x > 0.5:
            return "High"
        elif x > 0.2:
            return "Medium"
        else:
            return "Low"

    df["risk"] = ratio.apply(classify_risk)

    features = ["porcentaje_led", "total", "led", "mes_num", "localidad_enc"]
    df = df.dropna(subset=features + ["risk"])

    return df, features


# =========================
# GRAPHICS
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
def run_light():

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

    # training with EVAL for learning curve
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_train_sc, y_train), (X_test_sc, y_test)],
        eval_metric="multi_logloss"
    )

    y_pred      = model.predict(X_test_sc)
    y_proba     = model.predict_proba(X_test_sc)

    # =========================
    # METRICS
    # =========================
    accuracy  = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3)
    recall    = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3)
    f1        = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42, verbose=-1
        ))
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
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    confusion_graph = fig_to_base64()

    # =========================
    # 2. FEATURE IMPORTANCE (gain)
    # =========================
    fig, ax = plt.subplots(figsize=(7, 4))
    importance_gain = model.booster_.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance_gain)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(
        [features[i] for i in sorted_idx],
        importance_gain[sorted_idx],
        color=colors
    )
    ax.set_title("Feature Importance (Gain)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total profit")
    feature_graph = fig_to_base64()

    # =========================
    # 3. LEARNING CURVE
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    results = model.evals_result_
    train_loss = results["training"]["multi_logloss"]
    test_loss  = results["valid_1"]["multi_logloss"]
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Train loss", color="#3498db", linewidth=2)
    ax.plot(epochs, test_loss,  label="Test loss",  color="#e74c3c", linewidth=2, linestyle="--")
    ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Log Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    learning_graph = fig_to_base64()

    # =========================
    # 4. RISK DISTRIBUTION 
    # =========================
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["risk"].value_counts().reindex(["High", "Medium", "Low"])
    bar_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white")
    ax.set_title("Risk Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Quantity")
    plt.xticks(rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    risk_graph = fig_to_base64()

    # =========================
    # 5. PREDICTION PROBABILITIES
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    class_labels = le.classes_
    colors_prob = ["#e74c3c", "#f39c12", "#2ecc71"]
    for i, (cls, color) in enumerate(zip(class_labels, colors_prob)):
        ax.hist(
            y_proba[:, i], bins=15, alpha=0.6,
            label=cls, color=color, edgecolor="white"
        )
    ax.set_title("Probability Distribution by Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    proba_graph = fig_to_base64()

    # insight
    max_idx     = np.argmax(importance_gain)
    top_feature = features[max_idx]
    top_value   = round(importance_gain[max_idx], 1)
    insight     = f"The most influential variable is '{top_feature}' with a profit of {top_value}"

    return {
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "cv":              cv,
        "confusion_graph": confusion_graph,
        "feature_graph":   feature_graph,
        "learning_graph":  learning_graph,
        "risk_graph":      risk_graph,
        "proba_graph":     proba_graph,
        "top_feature":     top_feature,
        "top_value":       top_value,
        "insight":         insight
    }


# =========================
# PREDICTION
# =========================
def predict_light(porcentaje_led, total, led, mes, localidad):

    df, features = load_data()

    X = df[features]
    y = df["risk"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    mes_order = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_num = mes_order.get(mes.lower(), 1)

    le_loc = LabelEncoder()
    le_loc.fit(df["localidad"].astype(str))
    try:
        localidad_enc = le_loc.transform([localidad])[0]
    except ValueError:
        localidad_enc = 0

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42, verbose=-1
        ))
    ])
    pipeline.fit(X, y_enc)

    sample = np.array([[porcentaje_led, total, led, mes_num, localidad_enc]])
    pred = pipeline.predict(sample)

    return le.inverse_transform(pred)[0]
