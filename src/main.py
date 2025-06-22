#%%
import polars as pl
from utils.graph import Grafo
from itertools import combinations
from heapq import heappush, heappop, heapify

def load_data(path):
    initial_df = pl.read_csv(path)
    df = initial_df.select('show_id', 'director', 'cast').drop_nulls()
    df = df.with_columns(
        pl.col("director")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars().str.to_uppercase())
        .alias("director"),
        pl.col("cast")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars().str.to_uppercase())
        .alias("cast")
    )
    return df

def dfs_order(grafo: Grafo, source_node, visited, stack):
    visited.add(source_node)
    for adj, _ in grafo.adj_list[source_node]:
        if adj not in visited:
            dfs_order(grafo, adj, visited, stack)
    stack.append(source_node)

def dfs_componente(grafo: Grafo, source_node, visited, componente):
    visited.add(source_node)
    componente.append(source_node)
    for adj, _ in grafo.adj_list[source_node]:
        if adj not in visited:
            dfs_componente(grafo, adj, visited, componente)

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

# Algoritmos de centralidade

def dijkstra(grafo, origem):
    distancias = {v: float("inf") for v in grafo.adj_list}
    distancias[origem] = 0
    fila = [(0, origem)]
    heapify(fila)
    while fila:
        dist, atual = heappop(fila)
        for vizinho, peso in grafo.adj_list[atual]:
            nova_dist = dist + peso
            if nova_dist < distancias[vizinho]:
                distancias[vizinho] = nova_dist
                heappush(fila, (nova_dist, vizinho))
    return distancias

def degree_centrality(grafo, vertice):
    return len(grafo.adj_list.get(vertice, [])) / (len(grafo.adj_list) - 1) if len(grafo.adj_list) > 1 else 0

def closeness_centrality(grafo, vertice):
    dist = dijkstra(grafo, vertice)
    soma = sum(d for d in dist.values() if d != float("inf") and d > 0)
    if soma == 0:
        return 0
    return (len(dist) - 1) / soma

def betweenness_centrality(grafo):
    centralidade = {v: 0 for v in grafo.adj_list}
    vertices = list(grafo.adj_list.keys())
    for s in vertices:
        dist = dijkstra(grafo, s)
        for t in vertices:
            if s == t or dist[t] == float("inf"):
                continue
            caminho = [t]
            atual = t
            while atual != s:
                anterior = min(
                    (n for n, _ in grafo.adj_list[atual]),
                    key=lambda n: dist.get(n, float("inf")),
                    default=None
                )
                if anterior is None or dist[anterior] >= dist[atual]:
                    break
                caminho.append(anterior)
                atual = anterior
            for nodo in caminho[1:-1]:
                centralidade[nodo] += 1
    normalizador = ((len(vertices) - 1) * (len(vertices) - 2)) / 2
    for v in centralidade:
        centralidade[v] /= normalizador if normalizador else 1
    return centralidade

# %%
def main():
    df = load_data("netflix-data-graph-analysis/data/netflix_amazon_disney_titles.csv")
    print(df.head())

    df_exploded = df.explode(columns=['director']).explode(columns='cast')
    grafo_direcionado = Grafo(direcionado=True)
    for row in df_exploded.iter_rows(named=True):
        grafo_direcionado.adiciona_aresta(row['cast'], row['director'], peso=1)
    print(f"Vértices (direcionado): {grafo_direcionado.ordem}")
    print(f"Arestas (direcionado): {grafo_direcionado.tamanho}")

    grafo_nao_direcionado = Grafo(direcionado=False)
    for row in df.iter_rows(named=True):
        elenco = row["cast"]
        if len(elenco) < 2:
            continue
        for i in range(len(elenco)):
            for j in range(i + 1, len(elenco)):
                grafo_nao_direcionado.adiciona_aresta(elenco[i], elenco[j], peso=1)
    print(f"Vértices (não-direcionado): {grafo_nao_direcionado.ordem}")
    print(f"Arestas (não-direcionado): {grafo_nao_direcionado.tamanho}")

    componentes = kosaraju(grafo=grafo_direcionado)
    print(f"Número de componentes fortemente conexas: {len(componentes)}")
    print("Componentes com mais de 1 vértice:")
    for componente in componentes:
        if len(componente) > 1:
            print(componente)

    # CENTRALIDADE: GRAFO DIRECIONADO
    print("\n--- Centralidade no Grafo Direcionado ---")
    graus = {v: degree_centrality(grafo_direcionado, v) for v in grafo_direcionado.adj_list}
    print("Top 10 Grau:")
    for v, val in sorted(graus.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    proximidade = {v: closeness_centrality(grafo_direcionado, v) for v in list(grafo_direcionado.adj_list)[:100]}
    print("\nTop 10 Proximidade:")
    for v, val in sorted(proximidade.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    intermed = betweenness_centrality(grafo_direcionado)
    print("\nTop 10 Intermediação:")
    for v, val in sorted(intermed.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    # CENTRALIDADE: GRAFO NÃO-DIRECIONADO
    print("\n--- Centralidade no Grafo Não-Direcionado ---")
    graus_nd = {v: degree_centrality(grafo_nao_direcionado, v) for v in grafo_nao_direcionado.adj_list}
    print("Top 10 Grau:")
    for v, val in sorted(graus_nd.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    proximidade_nd = {v: closeness_centrality(grafo_nao_direcionado, v) for v in list(grafo_nao_direcionado.adj_list)[:100]}
    print("\nTop 10 Proximidade:")
    for v, val in sorted(proximidade_nd.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

    intermed_nd = betweenness_centrality(grafo_nao_direcionado)
    print("\nTop 10 Intermediação:")
    for v, val in sorted(intermed_nd.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{v}: {val:.4f}")

# %%
main()
# %%
