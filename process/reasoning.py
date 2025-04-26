import json
from openai import OpenAI
import igraph as ig
from collections import deque
import random

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

def call_llm(template):
    """
    通用大模型调用封装：
    - 输入：template（prompt字符串）
    - 输出：LLM完整返回的文本内容
    """
    client = OpenAI(
        api_key="sk-xhUZwtWJmekrtdX2hLvnC6nnuNSfe6qNIidWbzRIQBoZCEMa",
        base_url="https://api.chatanywhere.tech/v1"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": template}]
    )

    answer = response.choices[0].message.content.strip()
    return answer

def bfs_igraph_multi_start(graph, start_vertices):
    """
    支持多个起点的 BFS，记录所有完整路径
    :param graph: igraph.Graph 对象
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
            print(f"node {node} 随机选择 {len(selected_neighbors)} 个邻居，选择前 {neighbors}，选择后 {selected_neighbors}")
            for neighbor in selected_neighbors:
                visited.add(neighbor)
                queue.append(neighbor)
                paths[neighbor] = paths[node] + [neighbor]  # 更新路径

    print(f"\n最终完整路径集合: {final_paths}")
    return final_paths


# def llm_select_neighbors(current_node, candidate_neighbors, current_path):
#     """
#     模拟 LLM 决策：从候选邻居中选择要走的节点
#     :param current_node: 当前节点
#     :param candidate_neighbors: 邻居列表
#     :param current_path: 当前已经走的路径
#     :return: 选择的邻居列表
#     """
#     select_k = 2
#     selected = random.sample(candidate_neighbors, min(select_k, len(candidate_neighbors)))
#     return selected

def llm_select_neighbors(current_node, candidate_neighbors, current_path):
    """
    调用大模型 LLM 决策：从候选邻居中选择要走的节点
    :param current_node: 当前节点
    :param candidate_neighbors: 邻居列表
    :param current_path: 当前已走的路径
    :return: LLM 选择的邻居列表
    """

    # 拼接Prompt，清楚告诉LLM当前节点、路径和候选邻居
    template = (
        f"当前节点为：{current_node}\n"
        f"当前已走路径为：{current_path}\n"
        f"候选邻居节点为：{candidate_neighbors}\n"
        "请从候选邻居中选择你认为最优的节点（可选择多个），"
        "返回一个 Python 列表格式，例如：['B', 'C']。"
    )

    # 调用大模型
    response = call_llm(template)
    print(f"🧠 LLM选择邻居回复：{response}")

    # 简单处理 LLM 返回（假设LLM返回的是 Python 列表格式字符串）
    try:
        selected = eval(response)
        if isinstance(selected, list):
            return selected
    except Exception as e:
        print(f"⚠️ LLM返回无法解析，默认随机选：{e}")

    # 如果 LLM 返回有误，fallback 到随机
    select_k = 2
    return random.sample(candidate_neighbors, min(select_k, len(candidate_neighbors)))


def llm_should_stop(final_paths):
    """
    调用大模型 LLM 判断：根据当前完整路径集合，决定是否停止BFS
    大模型会基于以下规则作答：
    - 如果路径数量超过3条，建议停止
    - 如果路径中包含关键节点 'J'，建议停止
    """
    # 动态拼接路径列表到 prompt 中
    template = (
        f"以下是当前完整路径集合：{final_paths}。\n"
        "请判断：是否应该停止遍历？\n"
        "规则：如果路径数量超过3条 或 路径中包含关键节点 'J'，则建议停止。\n"
        "请直接回答：是 或 否。"
    )

    # 调用封装好的 LLM
    response = call_llm(template)
    print("🧠 大模型回复：", response)

    # 自动识别LLM回答
    if "是" in response or "yes" in response.lower():
        print("LLM判定：停止")
        return True
    else:
        print("LLM判定：继续搜索")
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
final_full_paths = bfs_igraph_multi_start(g, multi_start_nodes)

print("\n最终完整路径:")
for path in final_full_paths:
    print(path)