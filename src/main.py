import polars as pl
from utils.graph import Grafo
import heapq
from tqdm import tqdm
import numpy as np

def load_data(path):
    initial_df = pl.read_csv(path)
    df = initial_df.select("show_id", "director", "cast").drop_nulls()

    df = df.with_columns(
        pl.col("director")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars().str.to_uppercase())
        .alias("director"),
        pl.col("cast")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars().str.to_uppercase())
        .alias("cast"),
    )
    return df


def dfs_order(grafo: Grafo, source_node, visited, stack):
    visited.add(source_node)
    for adj, _ in grafo.adj_list[source_node]:
        if adj not in visited:
            dfs_order(grafo, adj, visited, stack)
    stack.append(source_node)


def dfs_componente(grafo: Grafo, source_node, visited, componente):
    stack = [source_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            componente.append(node)
            for adj, _ in grafo.adj_list[node]:
                if adj not in visited:
                    stack.append(adj)

def kosaraju(grafo: Grafo):
    if not grafo.direcionado:
        raise ValueError("Kosaraju funciona apenas com grafos direcionados.")
    visited = set()
    stack = []
    for vertice in grafo.adj_list.keys():
        if vertice not in visited:
            dfs_order(grafo, vertice, visited, stack)
    grafo_invertido = grafo.inverter_grafo()
    visited = set()
    componentes_fortemente_conexas = []
    while stack:
        vertice = stack.pop()
        if vertice not in visited:
            componente = []
            dfs_componente(grafo_invertido, vertice, visited, componente)
            componentes_fortemente_conexas.append(componente)
    return componentes_fortemente_conexas


def dijkstra(grafo, origem):
    distancias = {v: float("inf") for v in grafo.adj_list}
    distancias[origem] = 0
    visitados = set()
    fila = [(0, origem)]
    heapq.heapify(fila)
    while fila:
        dist, atual = heapq.heappop(fila)
        if atual in visitados:
            continue
        visitados.add(atual)
        for vizinho, peso in grafo.adj_list[atual]:
            nova_dist = dist + peso
            if nova_dist < distancias[vizinho]:
                distancias[vizinho] = nova_dist
                heapq.heappush(fila, (nova_dist, vizinho))
    return distancias


def degree_centrality(grafo, vertice):
    grau = len(grafo.adj_list.get(vertice, []))
    return grau / (len(grafo.adj_list) - 1) if len(grafo.adj_list) > 1 else 0


def closeness_centrality(grafo, vertice):
    dist = dijkstra(grafo, vertice)

    reachable_dists = [d for d in dist.values() if d != float("inf")]
    n_reachable = len(reachable_dists)

    if n_reachable <= 1:
        return 0.0

    soma_dist = sum(reachable_dists)
    if soma_dist == 0:
        return 0.0

    closeness = (n_reachable - 1) / soma_dist

    n_total = len(grafo.adj_list)
    if n_total > 1:
        normalization_factor = (n_reachable - 1) / (n_total - 1)
        normalized_closeness = closeness * normalization_factor
    else:
        normalized_closeness = 0.0

    return normalized_closeness

def agm_componente(self, x):
    """
    Retorna a AGM da componente que contém x, e o custo total.
    Só funciona em grafo não-direcionado.
    """
    if self.direcionado:
        raise ValueError("AGM só em grafo não-direcionado.")
    if x not in self.adj_list:
        raise KeyError(f"Vértice {x} não existe.")

    # 1) extrair componente conexa
    visited = set()
    componente = []
    # você já tem dfs_componente; se estiver aqui no módulo, chame-a:
    dfs_componente(self, x, visited, componente)

    # 2) Prim a partir de x
    visited = {x}
    heap = [(peso, x, v) for v, peso in self.adj_list[x]]
    heapq.heapify(heap)

    mst = []
    total = 0.0
    while heap and len(visited) < len(componente):
        peso, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        mst.append((u, v, peso))
        total += peso
        for w, pw in self.adj_list[v]:
            if w not in visited:
                heapq.heappush(heap, (pw, v, w))

    return mst, total

def centralidade_intermediacao(grafo: Grafo, u=None):
    """
    Calcula a centralidade de intermediação usando o algoritmo de Brandes.
    Esta é uma versão externa à classe Grafo.

    Args:
        grafo (Grafo): O objeto do grafo a ser analisado.
        u (any, optional): Se especificado, retorna a centralidade apenas para este vértice. Defaults to None.

    Returns:
        dict or float: Dicionário com a centralidade de todos os vértices ou o valor para o vértice 'u'.
    """
    bet = {v: 0.0 for v in grafo.adj_list}
    vertices = list(grafo.adj_list.keys())

    for fonte in tqdm(vertices, desc="Calculando Intermediação"):
        pilha = []
        pais = {v: [] for v in vertices}
        caminhos = {v: 0.0 for v in vertices}
        caminhos[fonte] = 1.0
        dist = {v: np.inf for v in vertices}
        dist[fonte] = 0
        fila = [(0, fonte)]

        while fila:
            d, v = heapq.heappop(fila)
            if d > dist[v]:
                continue
            pilha.append(v)
            for w, peso in grafo.adj_list.get(v, []):
                nova_dist = dist[v] + peso
                if nova_dist < dist[w]:
                    dist[w] = nova_dist
                    heapq.heappush(fila, (nova_dist, w))
                    caminhos[w] = caminhos[v]
                    pais[w] = [v]
                elif nova_dist == dist[w]:
                    caminhos[w] += caminhos[v]
                    pais[w].append(v)
        
        dep = {v: 0.0 for v in vertices}
        while pilha:
            w = pilha.pop()
            for v_p in pais[w]:
                if caminhos[w] != 0:
                    fracao = (caminhos[v_p] / caminhos[w]) * (1 + dep[w])
                    dep[v_p] += fracao
            if w != fonte:
                bet[w] += dep[w]

    n = grafo.ordem
    if n > 2:
        escala = 1 / ((n - 1) * (n - 2)) if grafo.direcionado else 2 / ((n - 1) * (n - 2))
        for v in bet:
            bet[v] *= escala

    if u is None:
        return bet
    if not grafo.tem_vertice(u):
        print(f"Vértice '{u}' não existe no grafo")
        return None
    return bet.get(u)

def contar_componentes_conexas(grafo: Grafo) -> int:
    if grafo.direcionado:
        raise ValueError("Esta função só deve ser usada com grafos não-direcionados.")

    visitados = set()
    componentes = 0

    for vertice in grafo.adj_list:
        if vertice not in visitados:
            stack = [vertice]
            while stack:
                atual = stack.pop()
                if atual not in visitados:
                    visitados.add(atual)
                    for vizinho, _ in grafo.adj_list[atual]:
                        if vizinho not in visitados:
                            stack.append(vizinho)
            componentes += 1

    return componentes


def main():
    # * Carregar dados
    df = load_data("data/netflix_amazon_disney_titles.csv")

    # * ex1 criação dos grafos
    df_exploded = df.explode(columns=["director"]).explode(columns="cast")
    grafo_direcionado = Grafo(direcionado=True)
    for row in df_exploded.iter_rows(named=True):
        grafo_direcionado.adiciona_aresta(row["cast"], row["director"], peso=1)
    print(f"Vértices (direcionado): {grafo_direcionado.ordem}")
    print(f"Arestas (direcionado): {grafo_direcionado.tamanho}")

    grafo_nao_direcionado = Grafo(direcionado=False)
    for row in df.iter_rows(named=True):
        elenco = row["cast"]
        if len(elenco) < 2:
            continue
        for i, ator_i in enumerate(elenco):
            for j in range(i + 1, len(elenco)):
                grafo_nao_direcionado.adiciona_aresta(ator_i, elenco[j], peso=1)

    print(f"Vértices (não-direcionado): {grafo_nao_direcionado.ordem}")
    print(f"Arestas (não-direcionado): {grafo_nao_direcionado.tamanho}")

    #* ex2 Componentes
    componentes = kosaraju(grafo=grafo_direcionado)
    print(f"Número de componentes fortemente conexas: {len(componentes)}")

    num_componentes = contar_componentes_conexas(grafo_nao_direcionado)
    print(
        f"\nNúmero de componentes conexas no grafo não-direcionado: {num_componentes}"
    )
    # CENTRALIDADE: GRAFO DIRECIONADO
    print("\n--- Centralidade no Grafo Direcionado ---")
    graus = {
        v: degree_centrality(grafo_direcionado, v) for v in grafo_direcionado.adj_list
    }
    print("Top 10 Grau:")
    for v, val in sorted(graus.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    proximidade = {
        v: closeness_centrality(grafo_direcionado, v)
        for v in list(grafo_direcionado.adj_list)[:100]
    }
    print("\nTop 10 Proximidade:")
    for v, val in sorted(proximidade.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    

    # CENTRALIDADE: GRAFO NÃO-DIRECIONADO
    print("\n--- Centralidade no Grafo Não-Direcionado ---")
    graus_nd = {
        v: degree_centrality(grafo_nao_direcionado, v)
        for v in grafo_nao_direcionado.adj_list
    }
    print("Top 10 Grau:")
    for v, val in sorted(graus_nd.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")
    proximidade_nd = {
    v: closeness_centrality(grafo_nao_direcionado, v)
    for v in list(grafo_nao_direcionado.adj_list)[:100]
}
    print("\nTop 10 Proximidade:")
    for v, val in sorted(proximidade_nd.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    intermed = centralidade_intermediacao(grafo_direcionado)
    print("\nTop 10 Intermediação:")
    for v, val in sorted(intermed.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    intermed_nd = centralidade_intermediacao(grafo_nao_direcionado)
    print("\nTop 10 Intermediação:")
    for v, val in sorted(intermed_nd.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

        
    # * ex3: AGM da componente contendo X
    X = next(iter(grafo_nao_direcionado.adj_list))  # ou defina X explicitamente
    _, mst_cost = agm_componente(grafo_nao_direcionado, X)
    print(f"\nAGM da componente contendo '{X}':")
    # for u, v, p in mst_edges:
    #     print(f"  {u} -- {v} (peso={p})")
    print(f"Custo total da AGM: {mst_cost}")
    


main()
