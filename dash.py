import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64
import numpy as np

from logistic import run_logistic
from rforest import run_forest
from lgbm_model import run_light
from xgboost_model import run_xgboost


def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return encoded


def run_dashboard():

    log_data = run_logistic()
    rf_data  = run_forest()
    lgb_data = run_light()
    xgb_data = run_xgboost()

    models = ["Logistic", "Random Forest", "LightGBM", "XGBoost"]
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

    metrics = {
        "accuracy":  [log_data["accuracy"],  rf_data["accuracy"],  lgb_data["accuracy"],  xgb_data["accuracy"]],
        "precision": [log_data["precision"], rf_data["precision"], lgb_data["precision"], xgb_data["precision"]],
        "recall":    [log_data["recall"],    rf_data["recall"],    lgb_data["recall"],    xgb_data["recall"]],
        "f1":        [log_data["f1"],        rf_data["f1"],        lgb_data["f1"],        xgb_data["f1"]],
        "cv":        [log_data["cv"],        rf_data["cv"],        lgb_data["cv"],        xgb_data["cv"]],
    }

    table_rows = []
    for i, m in enumerate(models):
        table_rows.append({
            "name":      m,
            "accuracy":  metrics["accuracy"][i],
            "precision": metrics["precision"][i],
            "recall":    metrics["recall"][i],
            "f1":        metrics["f1"][i],
            "cv":        metrics["cv"][i],
            "color":     colors[i],
        })

    best_idx   = int(np.argmax(metrics["f1"]))
    best_model = models[best_idx]
    best_f1    = metrics["f1"][best_idx]

    # =========================
    # 1. ACCURACY BAR
    # =========================
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(models, metrics["accuracy"], color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, metrics["accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy by Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    accuracy_graph = fig_to_base64()

    # =========================
    # 2. PRECISION / RECALL / F1 GROUPED BAR
    # =========================
    x     = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width, metrics["precision"], width, label="Precision", color="#a8dadc", zorder=3)
    b2 = ax.bar(x,         metrics["recall"],    width, label="Recall",    color="#457b9d", zorder=3)
    b3 = ax.bar(x + width, metrics["f1"],        width, label="F1-Score",  color="#e63946", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision, Recall and F1-Score by Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    metrics_graph = fig_to_base64()

    # =========================
    # 3. CROSS-VALIDATION HORIZONTAL BAR
    # =========================
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(models, metrics["cv"], color=colors, height=0.45, zorder=3)
    for i, val in enumerate(metrics["cv"]):
        ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_title("Cross-Validation Accuracy (5-Fold)", fontsize=14, fontweight="bold")
    ax.set_xlabel("CV Accuracy")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    cv_graph = fig_to_base64()

    # =========================
    # 4. RADAR CHART
    # =========================
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "CV"]
    num_vars     = len(metric_names)
    angles       = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles      += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (m, color) in enumerate(zip(models, colors)):
        values = [
            metrics["accuracy"][i],
            metrics["precision"][i],
            metrics["recall"][i],
            metrics["f1"][i],
            metrics["cv"][i],
        ]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=m)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Radar — Metrics Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    radar_graph = fig_to_base64()

    # =========================
    # 5. HEATMAP
    # =========================
    data_matrix = np.array([
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["cv"],
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data_matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=11)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for r in range(len(metric_names)):
        for c in range(len(models)):
            val = data_matrix[r, c]
            color_txt = "black" if val > 0.55 else "white"
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    color=color_txt, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title("Metrics Heatmap", fontsize=14, fontweight="bold", pad=30)
    plt.tight_layout()
    heatmap_graph = fig_to_base64()

    return {
        "table_rows":     table_rows,
        "best_model":     best_model,
        "best_f1":        best_f1,
        "models":         models,
        "metrics":        metrics,
        "accuracy_graph": accuracy_graph,
        "metrics_graph":  metrics_graph,
        "cv_graph":       cv_graph,
        "radar_graph":    radar_graph,
        "heatmap_graph":  heatmap_graph,
    }