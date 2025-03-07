import time
import torch
from torch_geometric import utils
import pandas as pd
import json
import numpy as np
from Theia.model import infer, GCN
from Theia.make_graph import add_attributes, prepare_graph
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

def Get_Adjacent(ids, mapp, edges, hops):
    if hops == 0:
        return set()

    neighbors = set()
    for edge in zip(edges[0], edges[1]):
        if any(mapp[node] in ids for node in edge):
            neighbors.update(mapp[node] for node in edge)

    if hops > 1:
        neighbors = neighbors.union(Get_Adjacent(neighbors, mapp, edges, hops - 1))

    return neighbors

def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0

    return prec, rec, fscore, FPR, TPR

# MP：模型预测的阳性集合（即模型预测为正例的项）。
# all_pids：所有可能的ID集合。
# GP：实际阳性集合（即真实的正例项）。
# edges：图中的边，表示元素之间的关系。
# mapp：一个映射，可能用于将项或节点映射到图中的特定元素或位置。
def helper(MP, all_pids, GP, edges, mapp):
    TP = MP.intersection(GP)
    FP = MP - GP
    FN = GP - MP
    TN = all_pids - (GP | MP)

    two_hop_gp = Get_Adjacent(GP, mapp, edges, 2)
    two_hop_tp = Get_Adjacent(TP, mapp, edges, 2)
    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp

    TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

    prec, rec, fscore, FPR, TPR = calculate_metrics(TP, FP, FN, TN)
    print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
    print(f"Precision: {round(prec, 2)}, Recall: {round(rec, 2)}, Fscore: {round(fscore, 2)}")

    return TPL, FPL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(30,5).to(device)

f = open("theia_test.txt")
data = f.read().split('\n')
data = [line.split('\t') for line in data]
df = pd.DataFrame (data, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
df = df.dropna()
df.sort_values(by='timestamp', ascending=True,inplace=True)
df = add_attributes(df,"ta1-theia-e3-official-6r.json.8")

with open("../data_files/theia.json", "r") as json_file:
    GT_mal = set(json.load(json_file))

data = df
phrases,labels,edges,mapp,relations,G = prepare_graph(data)
nodes = [infer(x) for x in phrases]
nodes = np.array(nodes)
et = time.time()

all_ids = list(data['actorID']) + list(data['objectID'])
all_ids = set(all_ids)

graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device), y=torch.tensor(labels, dtype=torch.long).to(device),
             edge_index=torch.tensor(edges, dtype=torch.long).to(device))
graph.n_id = torch.arange(graph.num_nodes)
flag = torch.tensor([True] * graph.num_nodes, dtype=torch.bool)

for m_n in range(20):
    model.load_state_dict(
        torch.load(f'../trained_weights/theia/lword2vec_gnn_theia{m_n}_E3.pth', map_location=torch.device('cpu')))
    loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=5000)
    for subg in loader:
        model.eval()
        out = model(subg.x, subg.edge_index)

        sorted, indices = out.sort(dim=1, descending=True)
        conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
        conf = (conf - conf.min()) / conf.max()

        pred = indices[:, 0]
        cond = (pred == subg.y) & (conf > 0.53)
        flag[subg.n_id[cond]] = torch.logical_and(flag[subg.n_id[cond]],
                                                  torch.tensor([False] * len(flag[subg.n_id[cond]]), dtype=torch.bool))

index = utils.mask_to_index(flag).tolist()
ids = set([mapp[x] for x in index])
alerts = helper(set(ids), set(all_ids), GT_mal, edges, mapp)
