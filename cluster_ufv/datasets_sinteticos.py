#!/usr/bin/env python3
"""Synthetic datasets and DBSCAN parameter helpers for the UFV cluster runner."""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import NearestNeighbors

SEED = 42
RUN_MODE = "cluster_gpu"

# =============================================================================
# MÓDULO: Geração de datasets sintéticos + sugestão automática de parâmetros
#         para DBSCAN (eps e minPts)
# =============================================================================


def normalize_minmax(X):
    """Normaliza cada coluna (feature) de X para o intervalo [0, 1]."""
    X = np.asarray(X, dtype=np.float32)
    mn = X.min(axis=0)              # menor valor de cada coluna
    mx = X.max(axis=0)              # maior valor de cada coluna
    denom = mx - mn                 # amplitude (range) de cada coluna
    denom[denom == 0.0] = 1.0       # evita divisao por zero se a coluna for constante
    return ((X - mn) / denom).astype(np.float32)


def _augmentar_2d_para_dim(X2, target_dim, rng, ruido=0.025):
    """
    Expande um dataset 2D (X2) para 'target_dim' colunas.
    Usado em datasets que só existem naturalmente em 2D (luas, circulos, espirais),
    para poder testar o DBSCAN tambem em dimensoes maiores.
    As colunas extras sao combinacoes (lineares e nao-lineares) das duas originais,
    com um pouco de ruido gaussiano somado para nao ficarem 100% redundantes.
    """
    X2 = np.asarray(X2, dtype=np.float32)
    target_dim = int(target_dim)

    if target_dim == 2:
        # nada a fazer, ja esta na dimensao pedida
        return X2

    extras = []
    x0, x1 = X2[:, 0], X2[:, 1]

    # "banco" de features derivadas de x0 e x1: combinacoes lineares,
    # funcoes trigonometricas, produtos e potencias
    candidatos = [
        0.70 * x0 + 0.15 * x1,
        -0.20 * x0 + 0.65 * x1,
        np.sin(x0),
        np.cos(x1),
        x0 * x1,
        x0 * x0,
        x1 * x1,
        x0 - x1,
    ]

    # para cada dimensao extra necessaria, pega um candidato do banco
    # (usa % para reciclar a lista caso precise de mais de 8 colunas extras)
    # e adiciona ruido gaussiano para nao ficar perfeitamente correlacionado
    for k in range(target_dim - 2):
        base = candidatos[k % len(candidatos)]
        extras.append(base + rng.normal(0.0, ruido, size=X2.shape[0]))

    # junta as 2 colunas originais + as colunas extras geradas
    return np.column_stack([X2] + extras).astype(np.float32)


def _dim_from_name(name):
    """
    Extrai a dimensao a partir do sufixo do nome do dataset.
    Ex: 'moons_10d' -> 10
    """
    m = re.search(r"_(\d+)d$", name)
    if not m:
        raise ValueError(f"Dataset sem sufixo de dimensao: {name}")
    return int(m.group(1))


def _centers_for_dim(dim, count=4, scale=4.0):
    """
    Gera 'count' centros de clusters em 'dim' dimensoes.
    A seed depende de dim e count, entao os centros sao reprodutiveis
    (mesmos parametros sempre geram os mesmos centros).
    Cada centro e "empurrado" em um eixo diferente para garantir
    que fiquem bem espalhados/separados uns dos outros.
    """
    rng_centers = np.random.default_rng(1000 + dim + count)
    centers = rng_centers.normal(0.0, scale, size=(count, dim)).astype(np.float32)
    for c in range(count):
        # alterna o sinal do deslocamento (+ para indices pares, - para impares)
        centers[c, c % dim] += scale * (1.0 if c % 2 == 0 else -1.0)
    return centers


def make_synthetic_dataset(name, n_samples, seed=SEED):
    """
    Catalogo de datasets sinteticos para testar clustering (DBSCAN).
    O nome do dataset (prefixo) decide qual "gerador" sera usado.
    Datasets 2D servem para validacao visual; os de dimensao maior (Nd)
    servem para benchmark de desempenho/robustez do algoritmo.
    """
    name = str(name).lower().strip()
    n_samples = int(n_samples)
    if n_samples < 8:
        raise ValueError("n_samples precisa ser >= 8")

    rng = np.random.default_rng(seed)
    dim = _dim_from_name(name)  # dimensao alvo extraida do nome
    visual = "validacao visual" if dim == 2 else "benchmark de desempenho"

    # -------------------------------------------------------------------
    # BLOBS DENSOS: 4 clusters bem compactos e bem separados
    # -------------------------------------------------------------------
    if name.startswith("dense_blobs_") and not name.startswith("dense_blobs_noise_"):
        centers = _centers_for_dim(dim, count=4, scale=4.5)
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=0.16,   # desvio padrao baixo = clusters bem compactos
            random_state=seed,
        )
        desc = f"Blobs densos {dim}D, baixa variancia intra-cluster ({visual})"
        obj = "Clusters densos e bem separados."

    # -------------------------------------------------------------------
    # BLOBS HETEROGENEOS: 3 grupos com densidades bem diferentes
    # (denso, medio, esparso) -- desafia o DBSCAN a lidar com um unico eps
    # -------------------------------------------------------------------
    elif name.startswith("heterogeneous_blobs_"):
        # divide o total de amostras em proporcoes fixas: 42%, 33% e o resto
        counts = [int(0.42 * n_samples), int(0.33 * n_samples)]
        counts.append(n_samples - sum(counts))
        centers = _centers_for_dim(dim, count=3, scale=4.5)
        stds = [0.08, 0.28, 0.72]  # desvios bem diferentes entre os 3 grupos

        parts, labels = [], []
        for idx, (count, std) in enumerate(zip(counts, stds)):
            Xi, _ = make_blobs(
                n_samples=count,
                centers=[centers[idx]],
                cluster_std=std,
                n_features=dim,
                random_state=seed + idx + 1,  # seed diferente por grupo
            )
            parts.append(Xi)
            labels.append(np.full(count, idx, dtype=np.int32))

        X = np.vstack(parts)
        y = np.concatenate(labels)
        desc = f"Blobs {dim}D com densidades diferentes ({visual})"
        obj = "Regioes densas, medias e esparsas no mesmo dataset."

    # -------------------------------------------------------------------
    # BLOBS DENSOS + RUIDO: blobs compactos + 20% de pontos de ruido
    # uniforme, rotulados como -1 (convencao para "outlier" em clustering)
    # -------------------------------------------------------------------
    elif name.startswith("dense_blobs_noise_"):
        n_noise = max(1, int(0.20 * n_samples))
        n_blob = n_samples - n_noise
        centers = _centers_for_dim(dim, count=4, scale=4.2)

        X_blob, y_blob = make_blobs(
            n_samples=n_blob,
            centers=centers,
            cluster_std=0.18,
            random_state=seed,
        )

        # gera ruido numa faixa maior que a ocupada pelos centros dos clusters
        low = float(np.min(centers) - 3.0)
        high = float(np.max(centers) + 3.0)
        noise = rng.uniform(low=low, high=high, size=(n_noise, dim))

        X = np.vstack([X_blob, noise])
        # rotulo -1 identifica os pontos de ruido (padrao usado pelo DBSCAN)
        y = np.concatenate([y_blob.astype(np.int32), np.full(n_noise, -1, dtype=np.int32)])
        desc = f"Blobs densos {dim}D com ruido/outliers rotulados como -1 ({visual})"
        obj = "Clusters densos com ruido fora das regioes principais."

    # -------------------------------------------------------------------
    # BLOBS SIMPLES: 6 centros, variancia moderada -- benchmark geral
    # -------------------------------------------------------------------
    elif name.startswith("blobs_"):
        X, y = make_blobs(
            n_samples=n_samples,
            centers=6,
            n_features=dim,
            cluster_std=0.45,
            random_state=seed,
        )
        desc = f"Blobs simples {dim}D, esfericos e variancia moderada ({visual})"
        obj = "Benchmark geral em diferentes dimensoes."

    # -------------------------------------------------------------------
    # MOONS: duas "luas" entrelacadas (formato nao-convexo classico)
    # gerado em 2D e depois expandido para a dimensao alvo
    # -------------------------------------------------------------------
    elif name.startswith("moons_"):
        X2, y = make_moons(n_samples=n_samples, noise=0.045, random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Duas luas com estrutura principal 2D em {dim}D ({visual})"
        obj = "Formato nao convexo preservado, com atributos extras em 10D quando aplicavel."

    # -------------------------------------------------------------------
    # RINGS: aneis/circulos concentricos (outro caso nao-convexo classico)
    # -------------------------------------------------------------------
    elif name.startswith("rings_"):
        X2, y = make_circles(n_samples=n_samples, factor=0.38, noise=0.025, random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Aneis concentricos com estrutura principal 2D em {dim}D ({visual})"
        obj = "Clusters nao convexos com buraco central."

    # -------------------------------------------------------------------
    # BLOBS ANISOTROPICOS: blobs normais em 2D "esticados"/rotacionados
    # por uma transformacao linear -- viram elipses em vez de circulos
    # -------------------------------------------------------------------
    elif name.startswith("anisotropic_blobs_"):
        X2, y = make_blobs(
            n_samples=n_samples,
            centers=[(-3, -1.5), (0, 2.2), (3, -1.0), (4.5, 2.5)],
            cluster_std=0.35,
            random_state=seed,
        )
        # matriz de transformacao linear que estica/rotaciona os pontos
        transform = np.array([[0.85, -0.55], [0.35, 1.35]], dtype=np.float32)
        X2 = X2 @ transform
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Blobs anisotropicos com estrutura principal 2D em {dim}D ({visual})"
        obj = "Clusters alongados por transformacao linear."

    # -------------------------------------------------------------------
    # BLOBS COM VARIANCIA VARIADA: 4 clusters, desvios bem diferentes
    # entre si (0.08 a 0.75) -- testa sensibilidade do eps
    # -------------------------------------------------------------------
    elif name.startswith("varied_blobs_"):
        X2, y = make_blobs(
            n_samples=n_samples,
            centers=[(-4, -2), (0, 2), (3.5, -1.5), (4, 3.2)],
            cluster_std=[0.08, 0.18, 0.42, 0.75],
            random_state=seed,
        )
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Blobs com variancia variada em {dim}D ({visual})"
        obj = "Sensibilidade de eps em clusters com raios diferentes."

    # -------------------------------------------------------------------
    # SPIRAL: 3 bracos espirais gerados manualmente via coordenadas
    # polares (theta cresce, raio cresce) -- clusters curvos e conectados
    # -------------------------------------------------------------------
    elif name.startswith("spiral_"):
        arms = 3
        counts = [n_samples // arms] * arms
        counts[-1] += n_samples - sum(counts)  # ajusta o resto da divisao no ultimo braco

        xs, ys = [], []
        for arm, count in enumerate(counts):
            # angulo cresce linearmente; cada braco comeca defasado (rotacionado)
            theta = np.linspace(0.35, 4.2 * np.pi, count, dtype=np.float32) + arm * (2.0 * np.pi / arms)
            # raio cresce linearmente do centro para fora
            radius = np.linspace(0.08, 1.0, count, dtype=np.float32)
            noise = rng.normal(0.0, 0.025, size=(count, 2)).astype(np.float32)
            # converte coordenadas polares (raio, theta) para cartesianas (x, y)
            pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]).astype(np.float32) + noise
            xs.append(pts)
            ys.append(np.full(count, arm, dtype=np.int32))

        X2 = np.vstack(xs)
        y = np.concatenate(ys)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Espirais com estrutura principal 2D em {dim}D ({visual})"
        obj = "Clusters curvos e conectividade por densidade."

    # -------------------------------------------------------------------
    # CHAIN BRIDGE: dois blocos separados, conectados por uma "ponte"
    # esparsa de pontos -- testa se o eps/minPts faz o DBSCAN "vazar"
    # e unir dois clusters que deveriam ser distintos
    # -------------------------------------------------------------------
    elif name.startswith("chain_bridge_"):
        n_bridge = max(8, int(0.16 * n_samples))
        n_left = (n_samples - n_bridge) // 2
        n_right = n_samples - n_bridge - n_left

        X_left, _ = make_blobs(n_samples=n_left, centers=[(-3.0, 0.0)], cluster_std=0.22, random_state=seed + 1)
        X_right, _ = make_blobs(n_samples=n_right, centers=[(3.0, 0.0)], cluster_std=0.22, random_state=seed + 2)

        # pontos da ponte: alinhados no eixo x, com um pouco de ruido no y
        bridge_x = np.linspace(-2.35, 2.35, n_bridge, dtype=np.float32)
        bridge_y = rng.normal(0.0, 0.055, size=n_bridge).astype(np.float32)
        X_bridge = np.column_stack([bridge_x, bridge_y])

        X2 = np.vstack([X_left, X_bridge, X_right])
        y = np.concatenate([
            np.zeros(n_left, dtype=np.int32),   # rotulo 0 = bloco esquerdo
            np.full(n_bridge, 2, dtype=np.int32),  # rotulo 2 = ponte
            np.ones(n_right, dtype=np.int32),   # rotulo 1 = bloco direito
        ])
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Dois blocos ligados por ponte esparsa em {dim}D ({visual})"
        obj = "Efeito de eps/minPts em conexao por cadeia."

    # -------------------------------------------------------------------
    # GRID BLOBS: grade 3x3 de micro-blobs bem compactos e proximos
    # -------------------------------------------------------------------
    elif name.startswith("grid_blobs_"):
        centers = [(x, y0) for x in (-3, 0, 3) for y0 in (-3, 0, 3)]  # 9 centros em grade
        X2, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.10, random_state=seed)
        X = _augmentar_2d_para_dim(X2, dim, rng)
        desc = f"Grade 3x3 de microblobs com estrutura principal 2D em {dim}D ({visual})"
        obj = "Muitos componentes pequenos e proximos."

    else:
        # nome nao corresponde a nenhum prefixo conhecido
        raise ValueError("Dataset nao cadastrado: " + name)

    return np.asarray(X, dtype=np.float32), y.astype(np.int32), desc, obj


def sample_rows(X, y, n_samples, seed=SEED):
    """Subamostra (sem reposicao) X e y para no maximo n_samples linhas."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = min(int(n_samples), len(X))
    rng = np.random.default_rng(seed)
    # se ja tiver poucas linhas, apenas usa todos os indices na ordem original
    idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
    return X[idx], y[idx]


def load_dataset_from_config(config, seed=SEED, verbose=True):
    """
    Funcao "maestro": le a config, gera o dataset, subamostra, normaliza
    e empacota tudo (dados + metadados) em um dicionario pronto para uso.
    """
    dataset_name = config["dataset_name"]
    n_samples = int(config["n_samples"])

    # 1) gera o dataset bruto (pode vir com mais linhas que n_samples, dependendo do gerador)
    X, y, descricao, objetivo_padrao = make_synthetic_dataset(dataset_name, n_samples, seed=seed)

    # 2) subamostra para garantir exatamente n_samples linhas
    #    (seed diferente da geracao, para nao correlacionar os dois processos aleatorios)
    X, y = sample_rows(X, y, n_samples, seed=seed + n_samples)

    # 3) normaliza cada feature para [0, 1]
    X = normalize_minmax(X)
    y = np.asarray(y, dtype=np.int32)

    loaded = {
        "dataset_name": dataset_name,
        "X": np.ascontiguousarray(X, dtype=np.float32),
        "y_true": y,
        "N": int(X.shape[0]),   # numero de amostras
        "D": int(X.shape[1]),   # numero de dimensoes/features
        "metadata": {
            "descricao": descricao,
            # se a config tiver um "objetivo" customizado, usa ele; senao usa o padrao do gerador
            "objetivo": config.get("objetivo", objetivo_padrao),
            "run_mode": RUN_MODE,
        },
    }

    if verbose:
        print("Dataset ativo:", loaded["dataset_name"])
        print("Shape:", loaded["X"].shape)
        print("Descricao:", descricao)
        print("Objetivo:", loaded["metadata"]["objetivo"])

    return loaded


def plot_dataset(X, y, title):
    """Plota um scatter 2D colorido por rotulo. So funciona se X tiver 2 colunas."""
    X = np.asarray(X)
    if X.shape[1] != 2:
        # nao ha como plotar diretamente em 2D se tiver mais de 2 dimensoes
        print(f"{title}: D={X.shape[1]} sem scatter 2D direto.")
        return
    plt.figure(figsize=(5.4, 4.4))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="tab10", linewidths=0)
    plt.title(title)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.tight_layout()


# =============================================================================
# SUGESTAO AUTOMATICA DE PARAMETROS PARA O DBSCAN (eps e minPts)
# =============================================================================


def _amostrar_linhas(X, max_pontos=5000, seed=SEED):
    """
    Subamostra X para no maximo 'max_pontos' linhas.
    Usado para acelerar calculos de vizinhos mais proximos em datasets grandes.
    """
    X = np.asarray(X, dtype=np.float32)
    max_pontos = int(max_pontos)
    if X.shape[0] <= max_pontos:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_pontos, replace=False)
    return np.ascontiguousarray(X[idx], dtype=np.float32)


def estimar_dimensao_intrinseca(X, k=20, sample_size=5000, seed=SEED):
    """Estima dimensao intrinseca por MLE de vizinhos proximos.

    A ideia e evitar escolher `minPts` apenas por dimensao ambiente. Para dados
    como luas/circulos, a dimensao intrinseca tende a ficar mais perto de 1 do
    que de 2, mas o tamanho do dataset ainda entra via log2(N).

    Implementa o estimador de Levina-Bickel: mede, para cada ponto, o quanto
    a distancia aos vizinhos cresce -- isso da uma pista de quantas dimensoes
    "reais" a estrutura dos dados ocupa (que pode ser bem menor que D).
    """
    Xs = _amostrar_linhas(X, sample_size, seed)
    k_eff = int(min(max(4, k), Xs.shape[0] - 1))  # garante k_eff entre 4 e N-1
    if k_eff < 3:
        # amostra pequena demais para estimar com confianca; usa a dimensao ambiente
        return float(X.shape[1])

    # busca os k_eff+1 vizinhos mais proximos (o +1 e porque o proprio ponto
    # sempre aparece como "vizinho" de distancia 0)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", n_jobs=-1)
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    dists = dists[:, 1:]  # remove o proprio ponto (distancia 0)

    rk = dists[:, -1]  # distancia ao vizinho mais distante (k-esimo vizinho)
    eps = 1e-12
    # formula do MLE de Levina-Bickel: log da razao entre a distancia maxima
    # e cada distancia intermediaria
    logs = np.log((rk[:, None] + eps) / (dists[:, :-1] + eps))
    inv_dim = np.mean(logs, axis=1)          # media por ponto = inverso da dimensao local
    dim_local = 1.0 / np.maximum(inv_dim, eps)  # inverte para obter a dimensao local

    dim = float(np.nanmedian(dim_local))  # mediana entre todos os pontos (robusta a outliers)
    # limita entre 1 (minimo possivel) e 2x a dimensao ambiente (evita valores absurdos)
    return float(np.clip(dim, 1.0, max(1.0, 2.0 * X.shape[1])))


def sugerir_minpts(X, sample_size=5000, minpts_max=256, seed=SEED):
    """Sugere candidatos de minPts usando dimensao intrinseca e log2(N)."""
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape

    dim_int = estimar_dimensao_intrinseca(X, sample_size=sample_size, seed=seed)

    # duas heuristicas classicas combinadas:
    base_dim = int(np.ceil(2.0 * dim_int))          # regra pratica: minPts ~ 2 * dimensao
    base_log = int(np.ceil(np.log2(max(n, 2))))      # minPts cresce com o tamanho do dataset
    base = max(4, base_dim, base_log)                # pega o maior valor (nunca menor que 4)

    # gera um leque de opcoes: da metade do "base" ate 4x o "base"
    candidatos = [base // 2, base, 2 * base, 4 * base]
    candidatos = sorted(set(int(np.clip(c, 4, minpts_max)) for c in candidatos))

    return np.asarray(candidatos, dtype=np.int32), dim_int


def sugerir_eps_por_knn(
    X,
    min_pts,
    quantis=(0.50, 0.70, 0.85),
    max_pontos=60000,
    seed=SEED,
    mostrar=True,
    titulo=None,
):
    """Sugere eps por quantis da curva k-distance.

    Como `minPts` inclui o proprio ponto nesta implementacao, usamos
    `n_neighbors=minPts`; o ultimo vizinho da matriz inclui a distancia ao
    minPts-esimo vizinho contando o proprio ponto como distancia zero.

    Essa e a versao automatizada da tecnica classica de "k-distance plot":
    em vez do usuario olhar o grafico e escolher visualmente o "cotovelo",
    o codigo sugere candidatos usando quantis da curva.
    """
    X = np.asarray(X, dtype=np.float32)
    Xs = _amostrar_linhas(X, max_pontos, seed)
    k_eff = int(min(max(2, int(min_pts)), Xs.shape[0]))

    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs)

    kth = np.sort(dists[:, -1])  # distancia de cada ponto ao seu k-esimo vizinho, ordenada
    # calcula os valores de eps candidatos como quantis dessa curva ordenada
    eps_values = np.asarray([np.quantile(kth, q) for q in quantis], dtype=np.float32)
    eps_values = np.unique(np.maximum(eps_values, np.float32(1e-6))).astype(np.float32)

    if mostrar:
        # plota a curva k-distance classica, com linhas tracejadas marcando os eps sugeridos
        plt.figure(figsize=(6.2, 3.8))
        plt.plot(kth)
        for eps in eps_values:
            plt.axhline(float(eps), linestyle="--", linewidth=1)
        plt.title(titulo or f"k-distance, minPts={min_pts}")
        plt.xlabel("pontos ordenados")
        plt.ylabel(f"distancia ao {k_eff}-esimo vizinho")
        plt.tight_layout()

    return eps_values


def sugerir_parametros_dbscan(X, quantis=(0.50, 0.70, 0.85), seed=SEED, titulo="", mostrar=None):
    """
    Funcao "maestro" da sugestao de parametros: combina minPts e eps
    em um unico fluxo, imprime um resumo e (opcionalmente) plota o grafico.
    """
    # 1) sugere candidatos de minPts a partir da dimensao intrinseca
    minpts_values, dim_int = sugerir_minpts(X, seed=seed)

    # usa o valor "do meio" da lista de candidatos como referencia para calcular o eps
    min_pts_ref = int(minpts_values[len(minpts_values) // 2])

    # so mostra o grafico automaticamente se os dados forem 2D (senao nao faz sentido visual)
    mostrar = (X.shape[1] == 2) if mostrar is None else bool(mostrar)

    # 2) sugere candidatos de eps usando o minPts de referencia
    eps_values = sugerir_eps_por_knn(
        X,
        min_pts_ref,
        quantis=quantis,
        seed=seed,
        mostrar=mostrar,
        titulo=titulo or f"k-distance para eps, minPts ref={min_pts_ref}",
    )

    print("dimensao intrinseca estimada:", round(dim_int, 3))
    print("minPts sugeridos:", minpts_values.tolist())
    print("minPts de referencia para eps:", min_pts_ref)
    print("eps sugeridos:", [float(v) for v in eps_values])

    return eps_values, minpts_values, min_pts_ref, dim_int
