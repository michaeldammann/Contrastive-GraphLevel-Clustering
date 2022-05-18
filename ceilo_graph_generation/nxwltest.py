from collections import Counter, defaultdict
from hashlib import blake2b
import networkx as nx
from networkx import weisfeiler_lehman_graph_hash as wlhash


def _hash_label(label, digest_size):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

def _init_node_labels(G, edge_attr, node_attr):
    if node_attr:
        return {u: str(dd[node_attr]) for u, dd in G.nodes(data=True)}
    elif edge_attr:
        return {u: "" for u in G}
    else:
        return {u: str(deg) for u, deg in G.degree()}


def _neighborhood_aggregate(G, node, node_labels, edge_attr=None):
    """
    Compute new labels for given node by aggregating
    the labels of each node's neighbors.
    """
    label_list = []
    for nbr in G.neighbors(node):
        prefix = "" if edge_attr is None else str(G[node][nbr][edge_attr])
        label_list.append(prefix + node_labels[nbr])
    return node_labels[node] + "".join(sorted(label_list))

def weisfeiler_lehman_graph_count(G, edge_attr=None, node_attr=None, iterations=3, digest_size=16):


    def weisfeiler_lehman_step(G, labels, edge_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            new_labels[node] = _hash_label(label, digest_size)
        return new_labels

    # set initial node labels
    node_labels = _init_node_labels(G, edge_attr, node_attr)

    subgraph_hash_counts = []
    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)
        counter = Counter(node_labels.values())
        # sort the counter, extend total counts
        subgraph_hash_counts.extend(sorted(counter.items(), key=lambda x: x[0]))

    # hash the final counter
    return sorted(counter.items(), key=lambda x: x[0])

def wl_kernel(G1, G2, iterations):
    g1_count = weisfeiler_lehman_graph_count(G1)
    g2_count = weisfeiler_lehman_graph_count(G2)



G1 = nx.Graph()
G1.add_nodes_from([0,1,2,3])
G1.add_edges_from([(0,1),(1,2),(2,3),(1,0),(2,1),(3,2)])

G2 = nx.Graph()
G2.add_nodes_from([0,1,2,3, 4])
G2.add_edges_from([(0,1),(1,2),(2,3),(1,0),(2,1),(3,2),(3,4), (4,3)])

print(weisfeiler_lehman_graph_count(G1, iterations=10))
print(weisfeiler_lehman_graph_count(G2, iterations=10))