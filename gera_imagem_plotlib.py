import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

caminho = Path(__file__).resolve().parent
df_map = pd.read_csv(caminho / "data" / "map_predito.csv")

id_to_cluster = {}
arquivos_cluster = sorted(glob.glob(str(caminho / "clusters_saida" / "cluster_*.txt")))

for idx, arq in enumerate(arquivos_cluster, start=1):
    with open(arq) as f:
        for line in f:
            line = line.strip()
            if line:
                id_to_cluster[int(line)] = idx

paredes = df_map[df_map["obst"] == 100]
obstaculos = df_map[(df_map["obst"] > 1.0) & (df_map["obst"] < 100)]
vitimas = df_map[df_map["id"] != -1].copy()
vitimas["cluster"] = vitimas["id"].map(id_to_cluster)

cores_tri = {0: "#2ca02c", 1: "#f1c40f", 2: "#e74c3c", 3: "#111111"}
cmap_clusters = plt.colormaps.get_cmap("tab20")

fig, ax = plt.subplots(figsize=(9, 9))

ax.scatter(obstaculos["x_rel"], obstaculos["y_rel"], c="#d3d3d3", marker="s", s=16, zorder=1)
ax.scatter(paredes["x_rel"], paredes["y_rel"], c="#1a1a1a", marker="s", s=20, label="Parede (obst=100)", zorder=2)

for cid in sorted(vitimas["cluster"].dropna().unique()):
    cid = int(cid)
    sub = vitimas[vitimas["cluster"] == cid]
    cor_c = cmap_clusters((cid - 1) % 20)

    ax.scatter(
        sub["x_rel"], sub["y_rel"],
        facecolors="none", edgecolors=cor_c,
        s=70, linewidths=1.8, zorder=3
    )

    ax.scatter(
        sub["x_rel"], sub["y_rel"],
        c=sub["tri"].map(cores_tri),
        s=18, zorder=4
    )

    cx, cy = sub["x_rel"].mean(), sub["y_rel"].mean()
    ax.text(
        cx, cy, f"C{cid}",
        fontsize=8, fontweight="bold", color="#000080",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=cor_c, alpha=0.85),
        zorder=5
    )

ax.scatter([0], [0], c="cyan", edgecolors="black", marker="*", s=180, label="Base (0, 0)", zorder=6)

handles_tri = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cores_tri[0], markersize=7, label="Tri 0: Verde"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cores_tri[1], markersize=7, label="Tri 1: Amarelo"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cores_tri[2], markersize=7, label="Tri 2: Vermelho"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cores_tri[3], markersize=7, label="Tri 3: Preto"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#1a1a1a", markersize=7, label="Parede (obst=100)"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="cyan", markeredgecolor="black", markersize=11, label="Base (0, 0)"),
]

ax.legend(handles=handles_tri, loc="lower left", fontsize=8.5, framealpha=0.9)
ax.set_title(f"Clusters K-Means (K={len(arquivos_cluster)}) com Triagem (tri)")
ax.set_xlabel("x_rel")
ax.set_ylabel("y_rel")
ax.grid(True, linestyle="-", alpha=0.25)

plt.tight_layout()
plt.savefig(caminho /"imagens" /"clusters_grid_plotlib.png", dpi=300)
plt.show()