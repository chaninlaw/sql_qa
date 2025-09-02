from langgraph.graph.state import CompiledStateGraph

def draw_graph(graph: CompiledStateGraph, file_path: str = "./assets/graph.png"):
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open(file_path, "wb") as f:
        f.write(png_bytes)