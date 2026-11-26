import itertools
import time
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42

caminho = Path(__file__).resolve().parent
CSV_MAPA = caminho / "data" / "map_predito.csv"
PASTA_ENTRADA = caminho / "clusters_saida"
PASTA_SAIDA = caminho / "clusters_sequenciados"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
SOBRESCREVER_ENTRADA = False

# --- Param Alg Genetico ---
TAM_POP = 120
GERACOES = 500
TAXA_CROSSOVER = 0.90
TAXA_MUTACAO = 0.30
TAM_TORNEIO = 3
ELITISMO = 4
N_EXECUCOES = 3
PACIENCIA = 120
BUSCA_LOCAL_A_CADA = 50
LIMITE_FORCA_BRUTA = 8

# Vermelho (2) = 0 | Amarelo (1) = 1 | Verde (0) = 2 | Preto (3) = 3
PRIO_MAP = {2: 0, 1: 1, 0: 2, 3: 3}
# Custo = soma de peso(prioridade) x distancia percorrida ate chegar na vitima
PESO_PRIO = {0: 50, 1: 10, 2: 1.0, 3: 0.1}
PESO_BASE = 1.0

# agentes
AGENTES_PODE_CRIANCA = {1: True, 2: True, 3: False, 4: False, 5: False}
TEMPO_POR_VITIMA = 1.0       # em unidades de dist


df = pd.read_csv(CSV_MAPA)
vitimas = df[df["id"] != -1].copy()
vitimas["id"] = vitimas["id"].astype(int)
vitimas["tri"] = vitimas["tri"].round().astype(int)
vitimas = vitimas.drop_duplicates("id").set_index("id")


class Problema:

    def __init__(self, ids):
        self.ids = list(ids)
        self.n = len(self.ids)
        sub = vitimas.loc[self.ids]
        coords = np.vstack([[0.0, 0.0], sub[["x_rel", "y_rel"]].to_numpy(float)])
        diff = coords[:, None, :] - coords[None, :, :]
        self.D = np.sqrt((diff ** 2).sum(-1))
        self.prios = np.array([PRIO_MAP[int(t)] for t in sub["tri"]])
        self.pesos = np.array([PESO_PRIO[int(p)] for p in self.prios])

    def trechos(self, M):
        r = M + 1
        m = M.shape[0]
        t = np.empty((m, self.n + 1))
        t[:, 0] = self.D[0, r[:, 0]]
        if self.n > 1:
            t[:, 1:self.n] = self.D[r[:, :-1], r[:, 1:]]
        t[:, self.n] = self.D[r[:, -1], 0]
        return t

    def custo_lote(self, M):
        M = np.atleast_2d(M)
        acum = np.cumsum(self.trechos(M), axis=1)
        w = np.concatenate([self.pesos[M], np.full((M.shape[0], 1), PESO_BASE)], axis=1)
        return (w * acum).sum(axis=1)

    def distancia(self, rota):
        return float(self.trechos(np.atleast_2d(rota)).sum())

    def inversoes(self, rota):
        c1 = c2 = c3 = inv = 0
        for idx in rota:
            p = self.prios[idx]
            if p == 0:
                inv += c1 + c2 + c3
            elif p == 1:
                inv += c2 + c3
                c1 += 1
            elif p == 2:
                inv += c3
                c2 += 1
            else:
                c3 += 1
        return int(inv)


#alg genetico
def construir_guloso(prob, rng, modo, k_rcl=1):
    
    restantes = list(range(prob.n))
    rota, atual = [], 0
    
    while restantes:
        if modo == "prio":
            pmin = min(prob.prios[i] for i in restantes)
            cand = [i for i in restantes if prob.prios[i] == pmin]
            score = [prob.D[atual, i + 1] for i in cand]
        else:
            cand = restantes
            score = [prob.D[atual, i + 1] / prob.pesos[i] for i in cand]
            
        ordem = np.argsort(score)[:k_rcl]
        pos = int(rng.choice(ordem)) if k_rcl > 1 else int(ordem[0])
        escolhido = cand[pos]
        rota.append(escolhido)
        restantes.remove(escolhido)
        atual = escolhido + 1
        
    return np.array(rota, dtype=int)


def bloco_aleatorio(prob, rng):
    rota = []
    for nivel in sorted(set(prob.prios.tolist())):
        grupo = np.flatnonzero(prob.prios == nivel)
        rota.extend(rng.permutation(grupo).tolist())
    return np.array(rota, dtype=int)


def inicializar(prob, rng):
    pop = [construir_guloso(prob, rng, "prio"), construir_guloso(prob, rng, "razao")]
    while len(pop) < TAM_POP:
        u = rng.random()
        if u < 0.35:
            pop.append(construir_guloso(prob, rng, "razao", 3))
        elif u < 0.65:
            pop.append(construir_guloso(prob, rng, "prio", 3))
        elif u < 0.85:
            pop.append(bloco_aleatorio(prob, rng))
        else:
            pop.append(rng.permutation(prob.n))
    return pop


def selecao_torneio(custos, rng):
    cand = rng.choice(len(custos), TAM_TORNEIO, replace=False)
    return int(cand[np.argmin(custos[cand])])


def order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    seg = p1[a:b + 1]
    p2r = np.roll(p2, -(b + 1))
    resto = p2r[~np.isin(p2r, seg)]
    filho = np.empty(n, dtype=int)
    filho[a:b + 1] = seg
    filho[[(b + 1 + k) % n for k in range(len(resto))]] = resto
    return filho


def mutar(ind, rng):
    n = len(ind)
    a, b = sorted(rng.choice(n, 2, replace=False))
    tipo = int(rng.integers(3))
    if tipo == 0:
        ind[a:b + 1] = ind[a:b + 1][::-1].copy()
    elif tipo == 1:
        ind[a], ind[b] = ind[b], ind[a]
    else:
        gene = ind[a]
        ind = np.insert(np.delete(ind, a), b, gene)
    return ind


def busca_local(prob, rota, max_passes=30):
    rota = rota.copy()
    melhor = float(prob.custo_lote(rota)[0])
    n = len(rota)
    for _ in range(max_passes):
        melhorou = False
        for i in range(n - 1):                       # 2-opt
            m = n - 1 - i
            cand = np.tile(rota, (m, 1))
            for k in range(m):
                j = i + 1 + k
                cand[k, i:j + 1] = rota[i:j + 1][::-1]
            custos = prob.custo_lote(cand)
            k = int(np.argmin(custos))
            if custos[k] < melhor - 1e-9:
                rota, melhor, melhorou = cand[k].copy(), float(custos[k]), True
        for i in range(n):                           # relocate
            no, base = rota[i], np.delete(rota, i)
            cand = np.array([np.insert(base, p, no) for p in range(n)])
            custos = prob.custo_lote(cand)
            k = int(np.argmin(custos))
            if custos[k] < melhor - 1e-9:
                rota, melhor, melhorou = cand[k].copy(), float(custos[k]), True
        if not melhorou:
            break
    return rota, melhor


def ag_uma_execucao(prob, rng):
    pop = inicializar(prob, rng)
    melhor_rota, melhor_custo, estag = None, float("inf"), 0

    for g in range(GERACOES):
        P = np.array(pop)
        custos = prob.custo_lote(P)

        if g % BUSCA_LOCAL_A_CADA == 0:
            i = int(np.argmin(custos))
            r, c = busca_local(prob, P[i])
            if c < custos[i]:
                P[i], custos[i] = r, c

        i = int(np.argmin(custos))
        if custos[i] < melhor_custo - 1e-9:
            melhor_custo, melhor_rota, estag = float(custos[i]), P[i].copy(), 0
        else:
            estag += 1
        if estag >= PACIENCIA:
            break

        taxa_mut = min(1.0, TAXA_MUTACAO * (2 if estag > PACIENCIA // 3 else 1))
        nova = [P[k].copy() for k in np.argsort(custos)[:ELITISMO]]
        while len(nova) < TAM_POP:
            p1 = P[selecao_torneio(custos, rng)]
            p2 = P[selecao_torneio(custos, rng)]
            if rng.random() < TAXA_CROSSOVER:
                f1, f2 = order_crossover(p1, p2, rng), order_crossover(p2, p1, rng)
            else:
                f1, f2 = p1.copy(), p2.copy()
            if rng.random() < taxa_mut:
                f1 = mutar(f1, rng)
            if rng.random() < taxa_mut:
                f2 = mutar(f2, rng)
            nova.append(f1)
            if len(nova) < TAM_POP:
                nova.append(f2)
        pop = nova

    return melhor_rota, melhor_custo


def sequenciar_cluster(ids, semente):
    prob = Problema(ids)
    n = prob.n

    if n == 1:
        rota = np.array([0])
    elif n <= LIMITE_FORCA_BRUTA:
        perms = np.array(list(itertools.permutations(range(n))))
        rota = perms[int(np.argmin(prob.custo_lote(perms)))]
    else:
        melhor_rota, melhor_custo = None, float("inf")
        for e in range(N_EXECUCOES):
            rng = np.random.default_rng(semente * 1000 + e)
            rota, custo = ag_uma_execucao(prob, rng)
            if custo < melhor_custo:
                melhor_rota, melhor_custo = rota, custo
        rota, _ = busca_local(prob, melhor_rota)

    return {
        "ids_ord": [prob.ids[i] for i in rota],
        "dist": prob.distancia(rota),
        "inv": prob.inversoes(rota),
    }


def atribuir(clusters):
    agentes = list(AGENTES_PODE_CRIANCA)
    custo_cl = [c["dist"] + TEMPO_POR_VITIMA * c["qtd"] for c in clusters]
    elegiveis = [
        [a for a in agentes if AGENTES_PODE_CRIANCA[a] or not c["tem_crianca"]]
        for c in clusters
    ]

    if len(clusters) <= 8:
        melhor, melhor_chave = None, None
        for combo in itertools.product(*elegiveis):
            cargas = dict.fromkeys(agentes, 0.0)
            for custo, a in zip(custo_cl, combo):
                cargas[a] += custo
            v = list(cargas.values())
            chave = (max(v), sum(x * x for x in v))
            if melhor_chave is None or chave < melhor_chave:
                melhor, melhor_chave = combo, chave
        return list(melhor)

    cargas = dict.fromkeys(agentes, 0.0)
    resultado = [None] * len(clusters)
    for i in sorted(range(len(clusters)), key=lambda i: -custo_cl[i]):
        a = min(elegiveis[i], key=lambda a: (cargas[a], a))
        cargas[a] += custo_cl[i]
        resultado[i] = a
    return resultado


def main():
    for f in PASTA_SAIDA.glob("cluster_*.txt"):
        f.unlink()

    arquivos = sorted(PASTA_ENTRADA.glob("cluster_*.txt"),
                      key=lambda p: int(p.stem.split("_")[1]))
    clusters = []
    t0 = time.time()
    for k, arq in enumerate(arquivos, start=1):
        ids = pd.read_csv(arq, header=None)[0].astype(int).tolist()
        sub = vitimas.loc[ids]
        res = sequenciar_cluster(ids, SEED + k)
        res.update(
            arquivo=arq.name,
            qtd=len(ids),
            tem_crianca=bool((sub["idade"] <= 12).any()),
            prio_media=float(sub["tri"].map(PRIO_MAP).mean()),
        )
        clusters.append(res)
        print(f"[{k}/{len(arquivos)}] {arq.name}: n={len(ids)} dist={res['dist']:.2f} "
              f"inversões={res['inv']} ({time.time() - t0:.1f}s)")

    agentes = atribuir(clusters)
    for c, a in zip(clusters, agentes):
        c["agente"] = a

    # ordem de execução por agente: clusters mais graves primeiro
    for a in AGENTES_PODE_CRIANCA:
        do_agente = sorted((c for c in clusters if c["agente"] == a),
                           key=lambda c: c["prio_media"])
        for ordem, c in enumerate(do_agente, start=1):
            c["ordem"] = ordem

    linhas = []
    for c in sorted(clusters, key=lambda c: (c["agente"], c["ordem"])):
        pd.Series(c["ids_ord"]).to_csv(PASTA_SAIDA / c["arquivo"], index=False, header=False)
        if SOBRESCREVER_ENTRADA:
            pd.Series(c["ids_ord"]).to_csv(PASTA_ENTRADA / c["arquivo"], index=False, header=False)
        sub = vitimas.loc[c["ids_ord"]]
        linhas.append({
            "Arquivo": c["arquivo"],
            "Agente": c["agente"],
            "Ordem": c["ordem"],
            "VD": int((sub["tri"] == 0).sum()),
            "AM": int((sub["tri"] == 1).sum()),
            "VM": int((sub["tri"] == 2).sum()),
            "PR": int((sub["tri"] == 3).sum()),
            "Total_Vitimas": c["qtd"],
            "Distancia_Trajeto": round(c["dist"], 2),
            "mu_sobr": round(float(sub["sobr"].mean()), 4),
            "sigma_sobr": round(float(sub["sobr"].std(ddof=1)), 4) if len(sub) > 1 else 0.0,
            "Inversoes_Prio": c["inv"],
        })

    res_df = pd.DataFrame(linhas)
    res_df.insert(0, "Seq", range(1, len(res_df) + 1))
    res_df.to_csv(PASTA_SAIDA / "tabela_resultados.csv", index=False)

    print("\n=== TABELA 1: RESULTADOS DE SOCORRO (SEQUENCIAMENTO VIA AG) ===")
    print(res_df.to_string(index=False))

    print("\n=== RESUMO FINAL (mu +/- sigma) ===")
    print(f"Total Vítimas    : {res_df['Total_Vitimas'].mean():.2f} +/- {res_df['Total_Vitimas'].std(ddof=1):.2f}")
    print(f"Distância Trajeto: {res_df['Distancia_Trajeto'].mean():.2f} +/- {res_df['Distancia_Trajeto'].std(ddof=1):.2f}")
    print(f"Soma das rotas   : {res_df['Distancia_Trajeto'].sum():.2f}")
    carga = res_df.groupby("Agente")["Distancia_Trajeto"].sum()
    print("Distância por agente:", {int(a): round(float(d), 2) for a, d in carga.items()})
    print(f"Arquivos salvos em: {PASTA_SAIDA}")

    try:                                             # figura
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        cores = {1: "tab:red", 2: "tab:orange", 3: "tab:green", 4: "tab:blue", 5: "tab:purple"}
        for c in clusters:
            sub = vitimas.loc[c["ids_ord"]]
            xs = [0.0] + sub["x_rel"].tolist() + [0.0]
            ys = [0.0] + sub["y_rel"].tolist() + [0.0]
            ax.plot(xs, ys, "-o", ms=3, lw=0.8, color=cores[c["agente"]],
                    label=f"{c['arquivo']} (A{c['agente']})")
        ax.plot(0, 0, "ks", ms=9, label="base")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        ax.set_title("Rotas sequenciadas")
        fig.savefig(PASTA_SAIDA / "rotas.png", dpi=150, bbox_inches="tight")
    except ImportError:
        pass


if __name__ == "__main__":
    main()