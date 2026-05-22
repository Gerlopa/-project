import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# =========================
# PATH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    BASE_DIR,
    "data",
    "datos luz.csv"
)

# =========================
# LOAD DATA
# =========================

def load_data():

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin1"
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df.rename(
        columns={
            "%ledxloca": "porcentaje_led"
        },
        inplace=True
    )

    # =========================
    # CLEAN PERCENTAGE
    # =========================

    df["porcentaje_led"] = (
        df["porcentaje_led"]
        .astype(str)
        .str.replace("%", "")
        .str.replace(",", ".")
    )

    df["porcentaje_led"] = pd.to_numeric(
        df["porcentaje_led"],
        errors="coerce"
    )

    # =========================
    # AVOID DIVISION BY ZERO
    # =========================

    df["total"] = df["total"].replace(0, 1)

    # =========================
    # CREATE RISK
    # =========================

    ratio = (
        df["mh"] + df["na"]
    ) / df["total"]

    def classify_risk(x):

        if x > 0.5:
            return "High"

        elif x > 0.2:
            return "Medium"

        return "Low"

    df["risk"] = ratio.apply(classify_risk)

    features = [
        "porcentaje_led",
        "total"
    ]

    df = df.dropna(
        subset=features + ["risk"]
    )

    return df, features

# =========================
# GRAPH HELPER
# =========================

def fig_to_base64():

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=80
    )

    buf.seek(0)

    encoded = base64.b64encode(
        buf.getvalue()
    ).decode()

    plt.close()

    return encoded

# =========================
# MAIN MODEL
# =========================

def run_forest():

    df, features = load_data()

    X = df[features]

    y = df["risk"]

    # =========================
    # ENCODE
    # =========================

    le = LabelEncoder()

    y_enc = le.fit_transform(y)

    # =========================
    # SPLIT
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    # =========================
    # SCALE
    # =========================

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)

    X_test_sc = scaler.transform(X_test)

    # =========================
    # MODEL
    # =========================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )

    model.fit(X_train_sc, y_train)

    # =========================
    # PREDICTIONS
    # =========================

    y_pred = model.predict(X_test_sc)

    y_proba = model.predict_proba(X_test_sc)

    # =========================
    # METRICS
    # =========================

    accuracy = round(
        accuracy_score(y_test, y_pred),
        3
    )

    precision = round(
        precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        3
    )

    recall = round(
        recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        3
    )

    f1 = round(
        f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        3
    )

    oob = round(
        model.oob_score_,
        3
    )

    # =========================
    # CROSS VALIDATION
    # =========================

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    cv = round(
        cross_val_score(
            pipeline,
            X,
            y_enc,
            cv=3,
            scoring="accuracy"
        ).mean(),
        3
    )

    # =========================
    # CONFUSION MATRIX
    # =========================

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=le.classes_,
        ax=ax,
        colorbar=False
    )

    ax.set_title("Confusion Matrix")

    confusion_graph = fig_to_base64()

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    fig, ax = plt.subplots(figsize=(5, 4))

    importance = model.feature_importances_

    ax.barh(
        features,
        importance
    )

    ax.set_title("Feature Importance")

    feature_graph = fig_to_base64()

    # =========================
    # PROBABILITY GRAPH
    # =========================

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, cls in enumerate(le.classes_):

        ax.hist(
            y_proba[:, i],
            bins=10,
            alpha=0.5,
            label=cls
        )

    ax.legend()

    ax.set_title(
        "Prediction Probabilities"
    )

    proba_graph = fig_to_base64()

    # =========================
    # RISK DISTRIBUTION
    # =========================

    fig, ax = plt.subplots(figsize=(5, 4))

    counts = df["risk"].value_counts()

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Risk Distribution"
    )

    risk_graph = fig_to_base64()

    # =========================
    # INSIGHT
    # =========================

    top_idx = np.argmax(importance)

    top_feature = features[top_idx]

    top_value = round(
        importance[top_idx],
        3
    )

    insight = (
        f"The most influential feature is "
        f"'{top_feature}' "
        f"with importance {top_value}"
    )

    # =========================
    # RETURN
    # =========================

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "cv": cv,

        "oob": oob,

        "confusion_graph": confusion_graph,

        "feature_graph": feature_graph,

        "proba_graph": proba_graph,

        "risk_graph": risk_graph,

        "insight": insight
    }

# =========================
# PREDICTION FUNCTION
# =========================

def predict_forest(
    porcentaje_led,
    total
):

    df, features = load_data()

    X = df[features]

    y = df["risk"]

    le = LabelEncoder()

    y_enc = le.fit_transform(y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    pipeline.fit(X, y_enc)

    sample = np.array([
        [porcentaje_led, total]
    ])

    pred = pipeline.predict(sample)

    return le.inverse_transform(pred)[0]
