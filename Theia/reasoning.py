import json
from openai import OpenAI

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="sk-xhUZwtWJmekrtdX2hLvnC6nnuNSfe6qNIidWbzRIQBoZCEMa",
#     base_url="https://api.chatanywhere.tech/v1"
#     # base_url="https://api.chatanywhere.org/v1"
# )
#
# response = client.chat.completions.create(
#     model="gpt-4",  # 选择模型
#     messages=[{"role": "user", "content": "你好，你是什么模型"}]
# )
#
# print(response.choices[0].message.content)

import igraph as ig
from collections import deque
import random

def bfs_igraph_multi_start(graph, start_vertices, select_k=2):
    """
    支持多个起点的 BFS，记录所有完整路径
    :param graph: igraph.Graph 对象
    :param start_vertices: 起始顶点的名称列表 (list)
    :param select_k: 每个节点随机选择的邻居数量
    :return: 所有完整路径 (list)
    """
    vertex_names = graph.vs["name"]
    paths = {}  # 记录每个节点的完整路径
    final_paths = []  # 存放完整路径结果

    # 构建邻接表
    adjacency_list = {name: [] for name in vertex_names}
    for edge in graph.es:
        source, target = vertex_names[edge.source], vertex_names[edge.target]
        adjacency_list[source].append(target)
        adjacency_list[target].append(source)

    # BFS 初始化
    visited = set()
    queue = deque()

    # 初始化多个起点
    print(f"初始节点{start_vertices}")
    for start in start_vertices:
        queue.append(start)
        visited.add(start)
        paths[start] = [start]

    while queue:
        node = queue.popleft()
        neighbors = [n for n in adjacency_list[node] if n not in visited]

        if not neighbors:
            # 叶子节点，记录完整路径
            final_paths.append("->".join(paths[node]))
        else:
            # 随机选择 K 个邻居扩展
            selected_neighbors = random.sample(neighbors, min(select_k, len(neighbors)))  # TODO 调 LLM 选择
            print(f"node {node} 随机选择 {select_k} 个邻居，选择前 {neighbors}，选择后 {selected_neighbors}")
            for neighbor in selected_neighbors:
                visited.add(neighbor)
                queue.append(neighbor)
                paths[neighbor] = paths[node] + [neighbor]  # 更新路径

    print(f"\n最终完整路径集合: {final_paths}")
    return final_paths

# 创建无向图
# g = ig.Graph(directed=False)
# g.add_vertices(["A", "B", "C", "D", "E", "F"])
# g.add_edges([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("E", "F")])

g = ig.Graph(directed=False)
g.add_vertices(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"])
g.add_edges([
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "E"), ("B", "F"),
    ("F", "J"),
    ("C", "G"),
    ("D", "H"), ("D", "I"),
    ("H", "K")
])

# 多个起点测试
multi_start_nodes = ["A", "C"]
final_full_paths = bfs_igraph_multi_start(g, multi_start_nodes, select_k=2)

print("\n最终完整路径:")
for path in final_full_paths:
    print(path)