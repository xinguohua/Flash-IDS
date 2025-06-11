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
from process.make_graph import add_attributes, prepare_graph, prepare_graph_new, collect_edges_from_log, \
    collect_nodes_from_log
from process.match.match import train_model
from process.model import EpochLogger, EpochSaver, GCN, infer
from process.partition import detect_communities
from embedding import graph_to_triples, train_embedding_model, get_feature_vector
from data import processor_factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = EpochLogger()
saver = EpochSaver()

# handler = processor_factory.get_handler("atlas")
handler = processor_factory.get_handler("darpa")


handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G  = handler.build_graph()

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
