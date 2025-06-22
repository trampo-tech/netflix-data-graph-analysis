import polars as pl
from utils.graph import Grafo
import heapq
import random
import collections
import multiprocessing
from functools import partial


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


def _brandes_centrality_subset(nodes_to_process, grafo):
    """Helper function for multiprocessing. Calculates centrality for a subset of nodes."""
    centrality = {v: 0.0 for v in grafo.adj_list}
    
    for s in nodes_to_process:
        # Single-source shortest paths (SSSP)
        S = []
        P = {w: [] for w in grafo.adj_list}
        sigma = {w: 0.0 for w in grafo.adj_list}
        sigma[s] = 1.0
        d = {w: -1 for w in grafo.adj_list}
        d[s] = 0
        
        Q = collections.deque([s])

        while Q:
            v = Q.popleft()
            S.append(v)
            # Assuming unweighted graph based on usage, peso=1
            for w, _ in grafo.adj_list.get(v, []):
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        
        # Accumulation
        delta = {w: 0.0 for w in grafo.adj_list}
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] != 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]
                
    return centrality

def betweenness_centrality(grafo, k=None, normalized=True, endpoints=False):
    """Brandes' algorithm for betweenness centrality, parallelized for CPU."""
    vertices = list(grafo.adj_list.keys())
    if k is not None:
        nodes_to_process = random.sample(vertices, k)
    else:
        nodes_to_process = vertices

    try:
        num_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        num_cores = 1 # Fallback for environments where cpu_count is not available

    if num_cores <= 1 or len(nodes_to_process) < 100: # Don't parallelize for small workloads
        # Run serially if only one core or small graph
        centrality = _brandes_centrality_subset(nodes_to_process, grafo)
    else:
        # Split the nodes to process among the cores
        chunk_size = max(1, len(nodes_to_process) // num_cores)
        node_chunks = [nodes_to_process[i:i + chunk_size] for i in range(0, len(nodes_to_process), chunk_size)]

        with multiprocessing.Pool(processes=num_cores) as pool:
            worker_func = partial(_brandes_centrality_subset, grafo=grafo)
            partial_centralities = pool.map(worker_func, node_chunks)

        # Aggregate the results from all processes
        centrality = {v: 0.0 for v in vertices}
        for partial_result in partial_centralities:
            for v, c in partial_result.items():
                centrality[v] += c

    # Normalization
    if normalized:
        n = len(vertices)
        if n < 2:
            return centrality
        
        if grafo.direcionado:
            scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
        else:
            scale = 1.0 / (((n - 1) * (n - 2)) / 2.0) if n > 2 else 1.0
            
        for v in centrality:
            centrality[v] *= scale

    return centrality

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

    intermed = betweenness_centrality(grafo_direcionado)
    print("\nTop 10 Intermediação:")
    for v, val in sorted(intermed.items(), key=lambda x: x[1], reverse=True)[:10]:
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

    intermed_nd = betweenness_centrality(grafo_nao_direcionado)
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
