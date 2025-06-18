from collections import defaultdict
import pickle
import heapq

class Grafo:
    def __init__(self, direcionado: bool):
        self.adj_list = defaultdict(list)
        self.direcionado = direcionado
        self.ordem = 0
        self.tamanho = 0

    def adiciona_vertice(self, u):
        if u not in self.adj_list:
            self.adj_list[u] = []
            self.ordem += 1

    def adiciona_aresta(self, u, v, peso : float):
        if peso < 0:
            print("Pesos negativos não são permitidos.")
            return

        if u not in self.adj_list:
            self.adiciona_vertice(u)
        if v not in self.adj_list:
            self.adiciona_vertice(v)

        for i, (vizinho, peso_atual) in enumerate(self.adj_list[u]):
            if vizinho == v:
                self.adj_list[u][i] = (v, peso + peso_atual)
                return

        self.adj_list[u].append((v, peso))
        self.tamanho += 1

    def remove_aresta(self, u, v):
        if self.tem_aresta(u, v):
            self.adj_list[u] = [(v2, p) for v2, p in self.adj_list[u] if v2 != v]
            self.tamanho -= 1

    def remove_vertice(self, u):
        if u in self.adj_list:
            # Diminui o tamanho com a quantidade de arestas que saem de u
            self.tamanho -= len(self.adj_list[u])
            
            # Remove o vértice
            del self.adj_list[u]
            self.ordem -= 1

            # Remove as arestas que chegam em u e atualiza o tamanho
            for vertice, vizinhos in self.adj_list.items():
                original_len = len(vizinhos)
                vizinhos[:] = [(v, p) for v, p in vizinhos if v != u]
                # Diminui o tamanho pela quantidade de arestas removidas
                self.tamanho -= (original_len - len(vizinhos)) 
 
   
    def tem_aresta(self, vertice1, vertice2):
        """
        Checks if there's an edge from node1 to node2
        returns a boolean
        """
        if vertice1 not in self.adj_list or vertice2 not in self.adj_list:
            return False
        if vertice2 in dict(self.adj_list[vertice1]):
            return True
        else:
            return False 

    def grau_entrada(self, u):
        return sum(
            1 for vizinhos in self.adj_list.values() if any(v == u for v, _ in vizinhos)
        )

    def grau_saida(self, u):
        return len(self.adj_list[u])

    def grau(self, u):
        return self.grau_entrada(u) + self.grau_saida(u)

    def get_peso(self, u, v):
        for vizinho, peso in self.adj_list[u]:
            if vizinho == v:
                return peso
        return None

    def maiores_graus_saida(self, num_lista=20):
        vertices = {}
        for u, _ in self.adj_list.items():
            saida_u = self.grau_saida(u)
            vertices[u] = saida_u

        top_n_vertices = heapq.nlargest(num_lista, vertices.keys(), key=lambda x: vertices[x])
        top_n_dict = {i: vertices[i] for i in top_n_vertices}
        return top_n_dict
    
    def maiores_graus_entrada(self, num_lista=20):
        #calcula os graus de entrada em uma passada
        in_degrees = defaultdict(int)
        for _, vizinhos in self.adj_list.items():
            for v, _ in vizinhos:
                in_degrees[v] += 1
        #retorna os n maiores
        top_n_vertices = heapq.nlargest(num_lista, in_degrees.items(), key=lambda x: x[1])

        return dict(top_n_vertices)

    def imprime_lista_adjacencias(self, str_return = False) -> str:
        lista = ""
        for u, vizinhos in self.adj_list.items():
            arestas = " -> ".join(f"({v}, {p})" for v, p in vizinhos)
            vertice = f"{u}: {arestas}"
            if not str_return:
                print(vertice)      
            lista += vertice

        if str_return:
            return lista
        else:
            return ""

    def pickle_graph(self, file_path: str):
        """Save this graph to a file using pickle."""
        with open(file_path, 'wb+') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    @classmethod
    def load_pickled_graph(cls, file_path: str):
        """Load a graph from a pickled file."""
        with open(file_path, 'rb+') as f:
            return pickle.load(f)
    