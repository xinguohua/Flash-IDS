# =================训练=========================
import numpy as np
import pandas as pd
import torch
import os
from gensim.models import Word2Vec
from sklearn.utils import class_weight
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from process.make_graph import add_attributes, prepare_graph, add_attributes_new
from process.match.match import train_model
from process.model import EpochLogger, EpochSaver, GCN, infer
from process.partition import detect_communities
from embedding import graph_to_triples, train_embedding_model, get_feature_vector


def collect_json_paths(base_dir):
    result = {}
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            result[subdir] = {"benign": [], "malicious": []}
            for category in ["benign", "malicious"]:
                category_path = os.path.join(subdir_path, category)
                if os.path.exists(category_path):
                    for file in os.listdir(category_path):
                        if file.endswith(".json") and not file.startswith("._"):
                            full_path = os.path.join(category_path, file)
                            result[subdir][category].append(full_path)
    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = EpochLogger()
saver = EpochSaver()

# 加载一个数据集
base_path = "../data_files/theia"
json_map = collect_json_paths(base_path)

all_dfs = []
for scene, category_data in json_map.items():
    for category, json_files in category_data.items():
        #  只处理良性类别
        if category != "benign":
            continue

        print(f"正在处理: 场景={scene}, 类别={category}, 文件={json_files}")
        scene_category = f"/{scene}_{category}.txt"
        f = open(base_path + scene_category)

        # 训练分隔
        data = f.read().split('\n')
        # data = [line.split('\t') for line in data]
        # for test
        data = [line.split('\t') for line in data[:1000]]
        df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
        df = df.dropna()
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        # 形成一个更完整的视图
        df = add_attributes_new(df, json_files)

        # 只取良性前80%训练
        num_rows = int(len(df) * 0.8)
        df = df.iloc[:num_rows]
        all_dfs.append(df)

# 训练用的数据集
benign_df = pd.concat(all_dfs, ignore_index=True)
benign_df = benign_df.drop_duplicates()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, labels, edges, mapp, relations, G = prepare_graph(benign_df)

# 大图分割
communities = detect_communities(G)

# 嵌入构造特征向量
# word2Vec
# word2vec = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=8, epochs=300,
#                     callbacks=[saver, logger])
# nodes = [infer(x) for x in phrases]
# nodes = np.array(nodes)
# TransE
nodes = []
triples = graph_to_triples(G, features, mapp)
trained_model, triples_factory = train_embedding_model(triples)
# mapp.index(G.vs[head_id]['name'])
nodeNames = [str(features[mapp.index(v["name"])]) for v in G.vs]  # 遍历 iGraph 的所有节点
node_embeddings = {node: get_feature_vector(trained_model, triples_factory, node) for node in nodeNames}
for node, embedding in node_embeddings.items():
    nodes.append(embedding)
    print(f"Node '{node}' embedding: {embedding[:5]}")  # 只显示前 5 维
nodes = np.array(nodes)

# 匹配模型
# 图神经网络输入构建
graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device), y=torch.tensor(labels, dtype=torch.long).to(device),
             edge_index=torch.tensor(edges, dtype=torch.long).to(device))
graph.n_id = torch.arange(graph.num_nodes)
mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool)

# 模型训练
# 匹配
train_model(G, communities)

# TODO：推理
