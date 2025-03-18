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
            # TODO：LLM选择
            if llm_should_stop(final_paths):
                print("LLM判定停止，BFS退出")
                break
        else:
            # 随机选择 K 个邻居扩展
            # ✅ TODO: LLM 控制选择策略（示例：LLM 让你选或筛选 neighbor）随机选择 K 个邻居扩展
            selected_neighbors = llm_select_neighbors(node, neighbors, paths[node])
            print(f"node {node} 随机选择 {select_k} 个邻居，选择前 {neighbors}，选择后 {selected_neighbors}")
            for neighbor in selected_neighbors:
                visited.add(neighbor)
                queue.append(neighbor)
                paths[neighbor] = paths[node] + [neighbor]  # 更新路径

    print(f"\n最终完整路径集合: {final_paths}")
    return final_paths


def llm_select_neighbors(current_node, candidate_neighbors, current_path):
    """
    模拟 LLM 决策：从候选邻居中选择要走的节点
    :param current_node: 当前节点
    :param candidate_neighbors: 邻居列表
    :param current_path: 当前已经走的路径
    :return: 选择的邻居列表
    """
    print(f"【LLM模拟】当前节点: {current_node}, 当前路径: {current_path}, 候选邻居: {candidate_neighbors}")

    select_k = 2
    selected = random.sample(candidate_neighbors, min(select_k, len(candidate_neighbors)))
    return selected


def llm_should_stop(final_paths):
    """
    模拟 LLM 判断：根据当前完整路径集合，决定是否停止BFS
    规则：
    - 如果完整路径数量达到3条，则停止
    - 或者路径中出现关键节点 J，也停止
    """
    print(f"🧠【LLM模拟】当前完整路径集合：{final_paths}")

    # 规则1：生成了3条完整路径，LLM决定够了
    if len(final_paths) >= 3:
        print("🧠【LLM模拟】路径数量达到3，停止！")
        return True

    # 规则2：只要有路径包含 'J'，LLM立刻决定停
    for path in final_paths:
        if 'J' in path:
            print("🧠【LLM模拟】命中关键节点J，停止！")
            return True

    return False

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