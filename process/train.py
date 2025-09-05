# =================训练=========================
import torch
from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.match.match import train_model
from process.partition import detect_communities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据集
# data_handler = get_handler("atlas")
# data_handler = get_handler("theia", True)
# data_handler = get_handler("cadets5", True)
data_handler = get_handler("optc", True)



# 加载数据
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G  = data_handler.build_graph()
# 大图分割
print("===============detect_communities=============")
communities = detect_communities(G)

# 嵌入构造特征向量
embedder_class = get_embedder_by_name("word2vec")
# embedder_class = get_embedder_by_name("transe")
embedder = embedder_class(G, features, mapp)
embedder.train()
node_embeddings = embedder.embed_nodes()
edge_embeddings = embedder.embed_edges()

# 模型训练
# 匹配
train_model(G, communities, node_embeddings, edge_embeddings)
