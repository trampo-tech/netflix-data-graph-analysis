#%%
import polars as pl
from utils.graph import Grafo
from itertools import combinations

def load_data(path):
    # load dataset and cleanup
    initial_df = pl.read_csv(path)
    # only care about id, director and cast
    df = initial_df.select('show_id', 'director', 'cast')\
                            .drop_nulls()

    # transform the strings into lists with the proper formatting
    df = df.with_columns(
        pl.col("director")
        .str.split(",")
        .list.eval(
            pl.element().str.strip_chars().str.to_uppercase()
        )
        .alias("director"),
        pl.col("cast")
        .str.split(",")
        .list.eval(
            pl.element().str.strip_chars().str.to_uppercase()
        )
        .alias("cast")
    )
    return df

def dfs_order(grafo: Grafo, source_node, visited, stack):
    """
    DFS que adiciona vértices à pilha na ordem de finalização.
    """
    visited.add(source_node)
    
    for adj, _ in grafo.adj_list[source_node]:
        if adj not in visited:
            dfs_order(grafo, adj, visited, stack)
    
    stack.append(source_node)

def dfs_componente(grafo: Grafo, source_node, visited, componente):
    """
    DFS que coleta todos os vértices de uma componente conexa.
    """
    visited.add(source_node)
    componente.append(source_node)
    
    for adj, _ in grafo.adj_list[source_node]:
        if adj not in visited:
            dfs_componente(grafo, adj, visited, componente)

def kosaraju(grafo: Grafo):
    """
    Implementa o algoritmo de Kosaraju para encontrar componentes fortemente conexas.
    
    Args:
        grafo (Grafo): Grafo direcionado.
    
    Returns:
        list: Lista de componentes fortemente conexas.
    """
    if not grafo.direcionado:
        raise ValueError("O algoritmo de Kosaraju funciona apenas com grafos direcionados.")
    
    # Passo 1: Executar DFS no grafo original e preencher pilha com ordem de finalização
    visited = set()
    stack = []
    
    for vertice in grafo.adj_list.keys():
        if vertice not in visited:
            dfs_order(grafo, vertice, visited, stack)
    
    # Passo 2: Criar grafo transposto (invertido)
    grafo_invertido = grafo.inverter_grafo()
    
    # Passo 3: Executar DFS no grafo transposto na ordem reversa da pilha
    visited = set()
    componentes_fortemente_conexas = []
    
    while stack:
        vertice = stack.pop()
        if vertice not in visited:
            componente = []
            dfs_componente(grafo_invertido, vertice, visited, componente)
            componentes_fortemente_conexas.append(componente)
    
    return componentes_fortemente_conexas 

# %% 
def main():

    df= load_data("data/netflix_amazon_disney_titles.csv")
    print(df.head())
    #* ex1: 
    df_exploded = df.explode(columns=['director']).explode(columns='cast')
    # every row is now a director -> cast in df_exploded

    grafo_direcionado = Grafo(direcionado=True)

    #criar o grafo direcionado
    for row in df_exploded.iter_rows(named=True):
        grafo_direcionado.adiciona_aresta(row['cast'], row['director'], peso=1)
    print(f"Quantidade de vértices do grafo direcionado: {grafo_direcionado.ordem}")
    print(f"Quantidade de arestas do grafo direcionado: {grafo_direcionado.tamanho}")


    grafo_nao_direcionado = Grafo(direcionado=False)

    for row in df.iter_rows(named=True):
        elenco = row["cast"]
        n = len(elenco)
        if n < 2:
            continue  

        for i in range(n):
            for j in range(i + 1, n):  
                ator1 = elenco[i]
                ator2 = elenco[j]
                grafo_nao_direcionado.adiciona_aresta(ator1, ator2, peso=1)

    print(f"Quantidade de vértices do grafo não-direcionado: {grafo_nao_direcionado.ordem}")
    print(f"Quantidade de arestas do grafo não-direcionado: {grafo_nao_direcionado.tamanho}")

    

    #* ex2:
    
    componentes = kosaraju(grafo=grafo_direcionado)
    print(f"Número de componente fortemente conexos: {len(componentes)}")
    # existem menos componentes que vertices, considerando a regra de criação desse grafo
    # significa que existem atores que foram diretores e diretores que foram atores. 
    print("Lista dos casos especiais:")
    for componente in componentes:
        if len(componente) > 1:
            print(componente)
    


    
#%%
main()
# %%
