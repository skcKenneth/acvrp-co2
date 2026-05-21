"""Tests for depot-reachable customer snapping."""
import networkx as nx

from src.data_loader import Customer, snap_customers_to_nodes


def _tiny_directed_graph() -> nx.MultiDiGraph:
    """
    Two components: {0,1} strongly connected (depot side) and {2,3} isolated.
    """
    g = nx.MultiDiGraph()
    for n, x, y in [(0, 0.0, 0.0), (1, 0.001, 0.0), (2, 10.0, 10.0), (3, 10.001, 10.0)]:
        g.add_node(n, x=x, y=y)
    g.add_edge(0, 1, length=100)
    g.add_edge(1, 0, length=100)
    g.add_edge(2, 3, length=50)
    g.graph["crs"] = "EPSG:4326"
    return g


def test_snap_respects_depot_strongly_connected_component():
    graph = _tiny_directed_graph()
    customers = [
        Customer(0, "Depot", 0.0, 0.0, 0),
        Customer(1, "NearDepot", 0.0005, 0.0, 5),
        # Geographically closer to node 2 (wrong SCC) than to node 1.
        Customer(2, "FarWrongSCC", 10.0005, 10.0, 3),
    ]
    node_ids = snap_customers_to_nodes(graph, customers, warn_on_resnap=False)
    reachable = {0, 1}
    assert node_ids[0] in reachable
    assert node_ids[1] in reachable
    assert node_ids[2] in reachable
