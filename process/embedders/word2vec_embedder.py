import numpy as np
import time, functools
from gensim.models import Word2Vec
from .base import GraphEmbedderBase
import os


# def graph_to_triples(G, features, mapp):
#     """
#     将 iGraph 图转换为 (头实体, 关系, 尾实体) 三元组
#     :param G: ig.Graph 实例
#     :return: list of triples (head, relation, tail)
#     """
#     triples = []
#     for edge in G.es:
#         head_id = edge.source  # 获取起点 ID
#         tail_id = edge.target  # 获取终点 ID
#         relation = edge['actions'] if 'actions' in edge.attributes() else "undefined_relation"  # 关系属性
#
#         head = str(features[mapp.index(G.vs[head_id]['name'])])
#         tail = str(features[mapp.index(G.vs[tail_id]['name'])])
#         triples.append([head, relation, tail])
#     return triples

def graph_to_triples(G, features, mapp):
    """
    将 iGraph 图转换为 (头实体, 关系, 尾实体) 三元组
    G: ig.Graph
    features: list，和 node_ids 顺序一致
    mapp: list 或 dict，node_id -> index
    """
    # 如果传进来是 list，就转成 dict，加速查找
    if not isinstance(mapp, dict):
        mapp = {name: i for i, name in enumerate(mapp)}

    # 预先把顶点的特征转成字符串，对齐到 igraph 的顶点索引
    feat_str_by_vid = [str(features[mapp[name]]) for name in G.vs["name"]]

    # 一次性拿出边和属性
    edges = G.get_edgelist()         # [(src, dst), ...]
    rels = (
        ["|".join(map(str, acts)) if isinstance(acts, (list, set)) else str(acts)
         for acts in G.es["action"]]
        if "action" in G.es.attributes()
        else ["undefined_relation"] * len(edges)
    )
    # 构建三元组
    triples = [
        [feat_str_by_vid[src], rels[i], feat_str_by_vid[dst]]
        for i, (src, dst) in enumerate(edges)
    ]
    return triples

class Word2VecEmbedder(GraphEmbedderBase):
    def __init__(self, G, features, mapp):
        super().__init__(G, features, mapp)
        self.model = None

    def train(self):
        phrases = graph_to_triples(self.G, self.features, self.mapp)
        print("graph_to_triples end")
        # self.model = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=4, epochs=100)
        self.model = Word2Vec(
    sentences=phrases,      # 你的语料，必须是 List[List[str]]
    vector_size=20,         # 词向量维度，调小更快
    window=3,               # 上下文窗口，调小减少计算
    min_count=5,            # 过滤低频词，减少词表大小
    workers=os.cpu_count()-1, # 用所有CPU核心并行
    epochs=3,               # 少迭代几次先试
    sg=0,                   # CBOW，比 skip-gram (sg=1) 快
    negative=5,             # 负采样个数
    sample=1e-3,            # 高频词子采样
    batch_words=20000,      # 大批量训练，提高吞吐
    sorted_vocab=0          # 构建词表时不排序，提速
)


    # def embed_nodes(self):
    #     node_embeddings = {}
    #     for v in self.G.vs:
    #         name = v["name"]
    #         try:
    #             phrase = self.features[self.mapp.index(name)]
    #             emb = self.model.wv[str(phrase)]
    #         except Exception:
    #             emb = np.zeros(30)
    #         node_embeddings[name] = emb
    #     return node_embeddings

    def embed_nodes(self):
        node_embeddings = {}
        if not isinstance(self.mapp, dict):
            name2idx = {n: i for i, n in enumerate(self.mapp)}
        else:
            name2idx = self.mapp

        for v in self.G.vs:
            name = v["name"]
            try:
                phrase = self.features[name2idx[name]]
                node_embeddings[name] = self.model.wv[str(phrase)]
            except KeyError:
                node_embeddings[name] = np.zeros(self.model.vector_size)
        return node_embeddings

    def embed_edges(self):
        edge_embeddings = {}
        for edge in self.G.es:
            if "action" in edge.attributes():
                acts = edge["action"]
                if isinstance(acts, (list, set)):
                    relation = "|".join(map(str, acts))  # 多个动作用 "|" 拼起来
                else:
                    relation = str(acts)  # 单个动作转字符串
            else:
                relation = "undefined_relation"
            if relation in self.model.wv:
                embedding = self.model.wv[relation]
            else:
                embedding = np.zeros(30)
            edge_embeddings[relation] = embedding
        return edge_embeddings

