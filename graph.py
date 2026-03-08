from graphviz import Digraph

# Recursively creates the graph
def build(value, nodes, edges):
    if value not in nodes:
        nodes.add(value)
        # Goes backwards to child nodes
        for child in value._prev:
            edges.add((child, value))
            build(child, nodes, edges)

def draw_graph(root):
    nodes, edges = set(), set()
    build(root, nodes, edges)

    graph = Digraph(format = "svg", graph_attr = {"rankdir": "LR"})

    for node in nodes:
        # Make sure nodes are unique
        uid = str(id(node))
        graph.node(name = uid, label = f"{node.label} | {node.v:.4f}", shape = "record")

        # If the node is the result of an operation, make an op node
        if node._op != "":
            graph.node(name = uid + node._op, label = node._op)
            # Connect that op node to the resulting node
            graph.edge(uid + node._op, uid) 

    # Connect all nodes to the operation that results in their parent nodes
    for node1, node2 in edges:
        graph.edge(str(id(node1)), str(id(node2)) + node2._op)

    return graph