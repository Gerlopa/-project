import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.pipeline import Pipeline

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

    mes_order = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    df["mes_num"] = df["mes"].str.lower().map(mes_order)

    le_loc = LabelEncoder()
    df["localidad_enc"] = le_loc.fit_transform(df["localidad"].astype(str))

    features = ["porcentaje_led", "total", "led", "mes_num", "localidad_enc"]
    df = df.dropna(subset=features)

    return df, features, le_loc


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
def run_clustering():

    df, features, le_loc = load_data()

    X = df[features]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # =========================
    # ELBOW + SILHOUETTE
    # para encontrar el K óptimo
    # =========================
    inertias    = []
    silhouettes = []
    k_range     = range(2, 9)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_sc, km.labels_))

    # K óptimo = mejor silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]

    # modelo final
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    model.fit(X_sc)
    labels = model.labels_

    df["cluster"] = labels

    silhouette = round(silhouette_score(X_sc, labels), 3)
    inertia    = round(model.inertia_, 1)

    # descripción de cada cluster
    cluster_summary = []
    for c in range(best_k):
        sub = df[df["cluster"] == c]
        cluster_summary.append({
            "id":             c,
            "n":              len(sub),
            "porcentaje_led": round(sub["porcentaje_led"].mean(), 1),
            "total":          round(sub["total"].mean(), 1),
            "led":            round(sub["led"].mean(), 1),
        })

    # =========================
    # 1. ELBOW CURVE
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_range, inertias, marker="o", color="#3498db", linewidth=2)
    ax.set_title("Método del Codo (Elbow)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Inercia")
    ax.axvline(x=best_k, color="#e74c3c", linestyle="--", alpha=0.7, label=f"K óptimo = {best_k}")
    ax.legend()
    ax.grid(alpha=0.3)
    elbow_graph = fig_to_base64()

    # =========================
    # 2. SILHOUETTE SCORE por K
    # =========================
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_range, silhouettes, marker="o", color="#9b59b6", linewidth=2)
    ax.set_title("Silhouette Score por K", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.axvline(x=best_k, color="#e74c3c", linestyle="--", alpha=0.7, label=f"K óptimo = {best_k}")
    ax.legend()
    ax.grid(alpha=0.3)
    silhouette_graph = fig_to_base64()

    # =========================
    # 3. CLUSTERS EN 2D (PCA)
    # =========================
    pca    = PCA(n_components=2)
    X_pca  = pca.fit_transform(X_sc)
    var_explained = round(sum(pca.explained_variance_ratio_) * 100, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))

    for c, color in zip(range(best_k), colors):
        mask = labels == c
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=f"Cluster {c} (n={mask.sum()})",
            color=color, alpha=0.7, s=60, edgecolors="white"
        )

    # centroides en PCA
    centroids_pca = pca.transform(model.cluster_centers_)
    ax.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        marker="X", s=200, color="black", zorder=5, label="Centroides"
    )

    ax.set_title(f"Clusters en 2D (PCA — {var_explained}% varianza)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend()
    ax.grid(alpha=0.3)
    pca_graph = fig_to_base64()

    # =========================
    # 4. DISTRIBUCIÓN DE CLUSTERS
    # =========================
    fig, ax = plt.subplots(figsize=(6, 4))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    colors_bar = plt.cm.Set2(np.linspace(0, 1, best_k))
    cluster_counts.plot(kind="bar", ax=ax, color=colors_bar, edgecolor="white")
    ax.set_title("Distribución de Registros por Cluster", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cantidad")
    plt.xticks(rotation=0)
    for i, v in enumerate(cluster_counts):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    dist_graph = fig_to_base64()

    # =========================
    # 5. DENDROGRAMA (jerárquico)
    # =========================
    fig, ax = plt.subplots(figsize=(10, 5))
    sample_idx = np.random.choice(len(X_sc), min(50, len(X_sc)), replace=False)
    Z = linkage(X_sc[sample_idx], method="ward")
    dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color="gray")
    ax.set_title("Dendrograma (muestra de 50 registros)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Índice de muestra")
    ax.set_ylabel("Distancia")
    ax.grid(axis="y", alpha=0.3)
    dendro_graph = fig_to_base64()

    insight = (
        f"Se encontraron {best_k} clusters óptimos con Silhouette Score de {silhouette}. "
        f"El cluster más grande tiene {max(cluster_counts)} registros."
    )

    return {
        "best_k":           best_k,
        "silhouette":       silhouette,
        "inertia":          inertia,
        "cluster_summary":  cluster_summary,
        "elbow_graph":      elbow_graph,
        "silhouette_graph": silhouette_graph,
        "pca_graph":        pca_graph,
        "dist_graph":       dist_graph,
        "dendro_graph":     dendro_graph,
        "insight":          insight
    }


# =========================
# PREDICTION
# =========================
def predict_cluster(porcentaje_led, total, led, mes, localidad):

    df, features, le_loc = load_data()

    X = df[features]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # K óptimo
    silhouettes = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        silhouettes.append(silhouette_score(X_sc, km.labels_))
    best_k = list(range(2, 9))[np.argmax(silhouettes)]

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    model.fit(X_sc)

    mes_order = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_num = mes_order.get(mes.lower(), 1)

    try:
        localidad_enc = le_loc.transform([localidad])[0]
    except ValueError:
        localidad_enc = 0

    sample = np.array([[porcentaje_led, total, led, mes_num, localidad_enc]])
    sample_sc = scaler.transform(sample)

    cluster = model.predict(sample_sc)[0]

    return int(cluster)