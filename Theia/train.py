# =================训练=========================
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.utils import class_weight
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from Theia.make_graph import add_attributes, prepare_graph
from Theia.model import EpochLogger, EpochSaver, GCN, infer
from Theia.partition import detect_communities
from embedding import graph_to_triples,train_embedding_model,get_feature_vector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = EpochLogger()
saver = EpochSaver()

# 加载数据
f = open("theia_train.txt")
data = f.read().split('\n')
# data = [line.split('\t') for line in data]
# for test
data = [line.split('\t') for line in data[:50]]
df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
df = df.dropna()
df.sort_values(by='timestamp', ascending=True, inplace=True)
# 形成一个更完整的视图
df = add_attributes(df, "ta1-theia-e3-official-1r.json")

# 成整个大图+捕捉特征语料+简化策略这里添加
phrases, labels, edges, mapp, relations, G = prepare_graph(df)

# 大图分割
communities = detect_communities(G)


# 构造特征向量
# word2vec = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=8, epochs=300,
#                     callbacks=[saver, logger])
# nodes = [infer(x) for x in phrases]
# nodes = np.array(nodes)

triples = graph_to_triples(G)
trained_model, triples_factory = train_embedding_model(triples)
nodes = [v["name"] for v in G.vs]  # 遍历 iGraph 的所有节点
node_embeddings = {node: get_feature_vector(trained_model, triples_factory, node) for node in nodes}
for node, embedding in node_embeddings.items():
    print(f"Node '{node}' embedding: {embedding[:5]}")  # 只显示前 5 维

# 图神经网络输入构建
graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device), y=torch.tensor(labels, dtype=torch.long).to(device),
             edge_index=torch.tensor(edges, dtype=torch.long).to(device))
graph.n_id = torch.arange(graph.num_nodes)
mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool)

# 损失函数
all_classes = np.array([0, 1, 2, 3, 4, 5])
existing_classes = np.unique(labels)
weights = class_weight.compute_class_weight('balanced', classes=existing_classes, y=np.array(labels))
full_weights = np.ones(len(all_classes))
for cls, weight in zip(existing_classes, weights):
    idx = np.where(all_classes == cls)[0]
    full_weights[idx] = weight
class_weights = torch.tensor(full_weights, dtype=torch.float).to(device)
criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')

# TODO：匹配模型
# 模型训练
model = GCN(30, 6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for m_n in range(20):
    loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=5000, input_nodes=mask)
    total_loss = 0
    for subg in loader:
        model.train()
        optimizer.zero_grad()
        out = model(subg.x, subg.edge_index)
        loss = criterion(out, subg.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * subg.batch_size
    mask_sum = mask.sum().item()
    if mask_sum > 0:
        print(total_loss / mask_sum)
    else:
        print("No active nodes remaining to calculate loss.")

    loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=5000, input_nodes=mask)
    for subg in loader:
        model.eval()
        out = model(subg.x, subg.edge_index)

        sorted, indices = out.sort(dim=1, descending=True)
        conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
        conf = (conf - conf.min()) / conf.max()

        pred = indices[:, 0]
        cond = (pred == subg.y) | (conf >= 0.9)
        mask[subg.n_id[cond]] = False

    torch.save(model.state_dict(), f'lword2vec_gnn_theia{m_n}_E3.pth')
    print(f'Model# {m_n}. {mask.sum().item()} nodes still misclassified \n')
