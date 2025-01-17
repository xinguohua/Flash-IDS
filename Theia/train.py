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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = EpochLogger()
saver = EpochSaver()

model = GCN(30, 5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

f = open("theia_train.txt")
data = f.read().split('\n')
data = [line.split('\t') for line in data]
df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
df = df.dropna()
df.sort_values(by='timestamp', ascending=True, inplace=True)
df = add_attributes(df, "ta1-theia-e3-official-1r.json")
phrases, labels, edges, mapp = prepare_graph(df)

word2vec = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=8, epochs=300,
                    callbacks=[saver, logger])
l = np.array(labels)
class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(l), y=l)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')

nodes = [infer(x) for x in phrases]
nodes = np.array(nodes)

graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device), y=torch.tensor(labels, dtype=torch.long).to(device),
             edge_index=torch.tensor(edges, dtype=torch.long).to(device))
graph.n_id = torch.arange(graph.num_nodes)
mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool)

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
    print(total_loss / mask.sum().item())

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
