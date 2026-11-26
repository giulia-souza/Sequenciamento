import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
N_INIT = 20

PESO_SOBR = 0.25

# Candidatos a (clusters pediatricos, clusters de adultos).
KC_CANDIDATOS = (1, 2, 3)
KA_CANDIDATOS = (3, 4, 5, 6)
MIN_CLUSTERS_TOTAL = 5      # garante pelo menos 1 cluster por agente

# Use None para escolher automaticamente pela silhueta ponderada
KC_FINAL = None
KA_FINAL = None

caminho_base = Path(__file__).resolve().parent
caminho_data = caminho_base / "data"
pasta_saida = caminho_base / "clusters_saida"


df = pd.read_csv(caminho_data / "vitimas_preditas.csv")
df = df[df["id"] != -1].drop_duplicates("id").reset_index(drop=True)
df["id"] = df["id"].astype(int)
df["dist_base"] = np.hypot(df["x_rel"], df["y_rel"])

#nova padronização pra melhorar a separação dos clusters, considerando a escala do mapa e a amplitude de sobr
xy = df[["x_rel", "y_rel"]].to_numpy(float)
escala = max(float(np.ptp(xy, axis=0).max()), 1.0)
sobr = df["sobr"].to_numpy(float)
amp = float(np.ptp(sobr))
sobr_n = (sobr - sobr.min()) / amp if amp > 0 else np.zeros(len(sobr))
X_all = np.column_stack([xy, PESO_SOBR * escala * sobr_n])

mask_crianca = (df["idade"] <= 12).to_numpy()
mask_adulto = ~mask_crianca
X_cri = X_all[mask_crianca]
X_adu = X_all[mask_adulto]
n_cri, n_adu, n_tot = len(X_cri), len(X_adu), len(X_all)

print(f"Crianças (<=12): {n_cri} | Maiores (>12): {n_adu}\n")


def ajustar(X, k):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=N_INIT)
    lab = km.fit_predict(X)
    sil = silhouette_score(X, lab) if 1 < k < len(X) else 0.0
    return km, lab, sil

kcs = [0] if n_cri == 0 else [k for k in KC_CANDIDATOS if k <= n_cri]
kas = [k for k in KA_CANDIDATOS if k <= n_adu]

fits_c = {k: ajustar(X_cri, k) for k in kcs if k > 0}
fits_a = {k: ajustar(X_adu, k) for k in kas}

print("=== Comparação de Parametrizações (Estratificado por Idade) ===")
resultados = []
for kc in kcs:
    for ka in kas:
        if kc > 0:
            km_c, lab_c, sil_c = fits_c[kc]
            in_c = km_c.inertia_
            tam_c = np.bincount(lab_c)
        else:
            in_c, sil_c, tam_c = 0.0, 0.0, np.array([], dtype=int)
        km_a, lab_a, sil_a = fits_a[ka]
        tamanhos = np.concatenate([tam_c, np.bincount(lab_a)])
        sil_pond = (sil_c * n_cri + sil_a * n_adu) / n_tot
        desb = tamanhos.max() / tamanhos.mean()
        resultados.append((kc, ka, sil_pond))
        print(
            f"Total={kc + ka} (Cri={kc}, Adu={ka}) | SSE={in_c + km_a.inertia_:10.2f} | "
            f"Sil_Cri={sil_c:.4f} | Sil_Adu={sil_a:.4f} | Sil_Pond={sil_pond:.4f} | "
            f"Desbalanceamento(max/media)={desb:.2f}"
        )

if KC_FINAL is None or KA_FINAL is None:
    elegiveis = [r for r in resultados if r[0] + r[1] >= MIN_CLUSTERS_TOTAL] or resultados
    KC_FINAL, KA_FINAL, _ = max(elegiveis, key=lambda r: r[2])
    print(f"\nEscolha automática (maior silhueta ponderada): Cri={KC_FINAL}, Adu={KA_FINAL}")
else:
    if KC_FINAL > 0 and KC_FINAL not in fits_c:
        fits_c[KC_FINAL] = ajustar(X_cri, KC_FINAL)
    if KA_FINAL not in fits_a:
        fits_a[KA_FINAL] = ajustar(X_adu, KA_FINAL)
    print(f"\nEscolha manual: Cri={KC_FINAL}, Adu={KA_FINAL}")


df["cluster"] = -1
if KC_FINAL > 0:
    df.loc[mask_crianca, "cluster"] = fits_c[KC_FINAL][1]
df.loc[mask_adulto, "cluster"] = fits_a[KA_FINAL][1] + KC_FINAL

os.makedirs(pasta_saida, exist_ok=True)
for arq in glob.glob(str(pasta_saida / "cluster_*.txt")):
    os.remove(arq)

resumo = []
for i, cid in enumerate(sorted(df["cluster"].unique()), start=1):
    sub = df[df["cluster"] == cid]
    with open(pasta_saida / f"cluster_{i}.txt", "w") as f:
        for vid in sub["id"]:
            f.write(f"{vid}\n")

    tem_crianca = bool((sub["idade"] <= 12).any())
    resumo.append({
        "Arquivo": f"cluster_{i}.txt",
        "Perfil": "Pediátrico (A1/A2)" if tem_crianca else "Adulto (A1-A5)",
        "Qtd_Vitimas": len(sub),
        "Criancas_<=12": int((sub["idade"] <= 12).sum()),
        "Idade_media": round(sub["idade"].mean(), 1),
        "sobr_media": round(sub["sobr"].mean(), 3),
        "Dist_media_base": round(sub["dist_base"].mean(), 1),
    })

resumo_df = pd.DataFrame(resumo)
resumo_df.to_csv(pasta_saida / "resumo_clusters.csv", index=False)
print(f"\n--- Resumo dos {KC_FINAL + KA_FINAL} clusters salvos em {pasta_saida} ---")
print(resumo_df.to_string(index=False))