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
from process.make_graph import add_attributes, prepare_graph, prepare_graph_new, add_attributes_new
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
        # TODO: for test
        if scene != "theia33":
            continue

        print(f"正在处理: 场景={scene}, 类别={category}, 文件={json_files}")
        scene_category = f"/{scene}_{category}.txt"
        f = open(base_path + scene_category)

        # 训练分隔
        data = f.read().split('\n')
        # TODO:
        # data = [line.split('\t') for line in data]
        # for test
        data = [line.split('\t') for line in data[:1000]]
        df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
        df = df.dropna()
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        # TODO 特征构建更完整
        # 形成一个更完整的视图
        df = add_attributes_new(df, json_files)

        # 只取良性前80%训练
        num_rows = int(len(df) * 0.8)
        df = df.iloc[:num_rows]
        all_dfs.append(df)

# 训练用的数据集
benign_df = pd.concat(all_dfs, ignore_index=True)
benign_df = benign_df.drop_duplicates()

benign_df.to_csv("benign_df.txt", sep='\t', index=False)

# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G = prepare_graph_new(benign_df)

# TODO 分割得检查下 为什么孤立的点得验证下 Community 0
# 大图分割
communities = detect_communities(G)

# 嵌入构造特征向量
# word2Vec
# word2vec = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=8, epochs=300,
#                     callbacks=[saver, logger])
# nodes = [infer(x) for x in phrases]
# nodes = np.array(nodes)
# TransE
triples = graph_to_triples(G, features, mapp)
trained_model, triples_factory = train_embedding_model(triples)
# 点
node_name_list = [v["name"] for v in G.vs]
node_feature_list = [str(features[mapp.index(name)]) for name in node_name_list]
node_embeddings = {}
for node_name, node_feature in zip(node_name_list, node_feature_list):
    embedding = get_feature_vector(trained_model, triples_factory, node_feature)
    node_embeddings[node_name] = embedding  # 使用node_name作为键存储embedding
    print(f"Node '{node_name}' embedding: {embedding[:5]}")  # 只显示前5维的embedding
# 边
edge_list = [edge['actions'] if 'actions' in edge.attributes() else "undefined_relation" for edge in G.es]
edge_embeddings = {}
for relation in edge_list:
    embedding = get_feature_vector(trained_model, triples_factory, relation)
    edge_embeddings[relation] = embedding
    print(f"Relation '{relation}' embedding: {embedding[:5]}")  # 打印前5维

# 模型训练
# 匹配
train_model(G, communities, node_embeddings, edge_embeddings)

# TODO：推理
