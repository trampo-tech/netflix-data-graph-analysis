from collections import defaultdict
import pickle
import heapq


class Grafo:
    """
    Classe para representar um grafo.

    Atributos:
        adj_list (defaultdict): Lista de adjacência do grafo.
        direcionado (bool): Indica se o grafo é direcionado.
        ordem (int): Ordem do grafo (número de vértices).
        tamanho (int): Tamanho do grafo (número de arestas).
    """

    def __init__(self, direcionado: bool):
        """
        Inicializa o grafo.

        Args:
            direcionado (bool): Indica se o grafo é direcionado.
        """
        self.adj_list = defaultdict(list)
        self.direcionado = direcionado
        self.ordem = 0
        self.tamanho = 0

    def adiciona_vertice(self, u):
        """
        Adiciona um vértice ao grafo.

        Args:
            u: Vértice a ser adicionado.
        """
        if u not in self.adj_list:
            self.adj_list[u] = []
            self.ordem += 1

    def adiciona_aresta(self, u, v, peso: float):
        """
        Adiciona uma aresta ao grafo.

        Args:
            u: Vértice de origem. Se
            v: Vértice de destino.
            peso (float): Peso da aresta.
        """
        if peso < 0:
            raise ValueError("Pesos negativos não são permitidos.")

        if u not in self.adj_list:
            self.adiciona_vertice(u)
        if v not in self.adj_list:
            self.adiciona_vertice(v)

        if not self.direcionado:
            for i, (vizinho, peso_atual) in enumerate(self.adj_list[v]):
                if vizinho == u:
                    self.adj_list[v][i] = (u, peso + peso_atual)
                    break
            else:
                self.adj_list[v].append((u, peso))

        for i, (vizinho, peso_atual) in enumerate(self.adj_list[u]):
            if vizinho == v:
                self.adj_list[u][i] = (v, peso + peso_atual)
                break
        else:
            self.adj_list[u].append((v, peso))
            self.tamanho += 1  # só adiciona ao tamanho uma unica vez

    def remove_aresta(self, u, v):
        """
        Remove uma aresta do grafo.

        Args:
            u: Vértice de origem.
            v: Vértice de destino.
        """
        if u != v and self.tem_aresta(u, v):
            self.adj_list[u] = [(v2, p) for v2, p in self.adj_list[u] if v2 != v]
            self.tamanho -= 1

        if u != v and not self.direcionado and self.tem_aresta(v, u):
            self.adj_list[v] = [(v2, p) for v2, p in self.adj_list[u] if v2 != u]

    def remove_vertice(self, u):
        """
        Remove um vértice do grafo.

        Args:
            u: Vértice a ser removido.
        """
        if u in self.adj_list:
            self.tamanho -= len(self.adj_list[u])
            del self.adj_list[u]
            self.ordem -= 1
            for _, vizinhos in self.adj_list.items():
                original_len = len(vizinhos)
                vizinhos[:] = [(v, p) for v, p in vizinhos if v != u]
                self.tamanho -= original_len - len(vizinhos)

    def tem_aresta(self, vertice1, vertice2):
        """
        Verifica se existe uma aresta de vertice1 para vertice2.

        Args:
            vertice1: Vértice de origem.
            vertice2: Vértice de destino.
        Returns:
            bool: True se existe a aresta, False caso contrário.
        """
        if vertice1 not in self.adj_list or vertice2 not in self.adj_list:
            return False
        if vertice2 in dict(self.adj_list[vertice1]):
            return True
        else:
            return False

    def grau_entrada(self, u):
        """
        Retorna o grau de entrada de um vértice.

        Args:
            u: Vértice desejado.
        Returns:
            int: Grau de entrada.
        """
        return sum(
            1 for vizinhos in self.adj_list.values() if any(v == u for v, _ in vizinhos)
        )

    def grau_saida(self, u):
        """
        Retorna o grau de saída de um vértice.

        Args:
            u: Vértice desejado.
        Returns:
            int: Grau de saída.
        """
        return len(self.adj_list[u])

    def grau(self, u):
        """
        Retorna o grau total de um vértice.

        Args:
            u: Vértice desejado.
        Returns:
            int: Grau total.
        """
        if self.direcionado:
            return self.grau_entrada(u) + self.grau_saida(u)
        else:
            return len(self.adj_list.get(u, []))

    def get_peso(self, u, v):
        """
        Retorna o peso de uma aresta.

        Args:
            u: Vértice de origem.
            v: Vértice de destino.
        Returns:
            float ou None: Peso da aresta ou None se não existir.
        """
        for vizinho, peso in self.adj_list[u]:
            if vizinho == v:
                return peso
        return None

    def maiores_graus_saida(self, num_lista=20):
        """
        Retorna os vértices com maiores graus de saída.

        Args:
            num_lista (int): Número de vértices a retornar.
        Returns:
            dict: Vértices e seus graus de saída.
        """
        vertices = {}
        for u, _ in self.adj_list.items():
            saida_u = self.grau_saida(u)
            vertices[u] = saida_u

        top_n_vertices = heapq.nlargest(
            num_lista, vertices.keys(), key=lambda x: vertices[x]
        )
        top_n_dict = {i: vertices[i] for i in top_n_vertices}
        return top_n_dict

    def maiores_graus_entrada(self, num_lista=20):
        """
        Retorna os vértices com maiores graus de entrada.

        Args:
            num_lista (int): Número de vértices a retornar.
        Returns:
            dict: Vértices e seus graus de entrada.
        """
        if not self.direcionado:
            # For undirected graphs, in-degree = out-degree = degree
            return self.maiores_graus_saida(num_lista)

        # Calculate in-degrees for directed graphs
        in_degrees = defaultdict(int)
        for _, vizinhos in self.adj_list.items():
            for v, _ in vizinhos:
                in_degrees[v] += 1

        top_n_vertices = heapq.nlargest(
            num_lista, in_degrees.items(), key=lambda x: x[1]
        )
        return dict(top_n_vertices)

    def imprime_lista_adjacencias(self, str_return=False) -> str:
        """
        Imprime a lista de adjacências do grafo.

        Args:
            str_return (bool): Se True, retorna a string ao invés de imprimir.
        Returns:
            str: Lista de adjacências como string se str_return for True, caso contrário string vazia.
        """
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

    def inverter_grafo(self):
        """
        Retorna um novo grafo com todas as arestas invertidas.

        Returns:
            Grafo: Novo grafo com arestas invertidas.
        """
        if not self.direcionado:
            raise ValueError("Não é possível inverter um grafo não direcionado.")

        grafo_invertido = Grafo(direcionado=True)

        # Adiciona todos os vértices primeiro
        for vertice in self.adj_list.keys():
            grafo_invertido.adiciona_vertice(vertice)

        # Inverte todas as arestas
        for u, vizinhos in self.adj_list.items():
            for v, peso in vizinhos:
                grafo_invertido.adiciona_aresta(u=v, v=u, peso=peso)

        return grafo_invertido

    def pickle_graph(self, file_path: str):
        """
        Salva o grafo em um arquivo usando pickle.

        Args:
            file_path (str): Caminho do arquivo para salvar.
        """
        with open(file_path, "wb+") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickled_graph(cls, file_path: str):
        """
        Carrega um grafo de um arquivo pickle.

        Args:
            file_path (str): Caminho do arquivo para carregar.
        Returns:
            Grafo: Instância do grafo carregado.
        """
        with open(file_path, "rb+") as f:
            return pickle.load(f)

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
