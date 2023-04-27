import itertools

import networkx as nx


def fully_connected_bipartite_graph(size_U: int, size_V: int) -> nx.Graph:
    B = nx.Graph()
    B.add_nodes_from([f"U_{u}" for u in range(size_U)], bipartite=0)
    B.add_nodes_from([f"V_{v}" for v in range(size_V)], bipartite=0)
    B.add_edges_from([
        (f"U_{u}", f"V_{v}") 
        for u, v in itertools.product(range(size_U), range(size_V))
    ])
    return B
