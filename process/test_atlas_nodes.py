# =================训练=========================
import os
import pandas as pd
import torch
from process.datahandlers.common import collect_dot_paths
from process.make_graph import prepare_graph_new, collect_edges_from_log, \
    collect_nodes_from_log
from process.match.test_model import test_model
from process.model import EpochLogger, EpochSaver
from process.partition import detect_communities
from datahandlers.atlas_handler import collect_nodes_from_log,collect_edges_from_log
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
all_dfs = []
all_netobj2pro = {}  # 网络对象 UUID → 属性字符串
all_subject2pro = {}  # 进程 UUID → 属性字符串
all_file2pro = {}  # 文件 UUID → 属性字符串

def merge_properties(src_dict, target_dict):
    for k, v in src_dict.items():
        if k not in target_dict:
            target_dict[k] = v
processed_data = []
domain_name_set = {}
ip_set = {}
connection_set = {}
session_set = {}
web_object_set = {}
# 处理每个 .dot 文件
# TODO test
dot_file=r"D:\数据集分析\Flash-IDS-main\atlas_data\graph_M1-CVE-2015-5122_windows_h1.dot"
print(f"正在处理文件: {dot_file}")

# 读取节点数据
netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set = collect_nodes_from_log(
        dot_file)

# 打印节点数据
    # 调用 `collect_edges_from_log` 收集边
df = collect_edges_from_log(dot_file, domain_name_set, ip_set, connection_set, session_set, web_object_set,
                                subject2pro, file2pro)  # 将 dot 文件传入收集边的函数
# 只取良性前90%训练
num_rows = int(len(df) * 0.9)
df = df.iloc[:num_rows]  # 正确缩进
all_dfs.append(df)
merge_properties(netobj2pro, all_netobj2pro)
merge_properties(subject2pro, all_subject2pro)
merge_properties(file2pro, all_file2pro)

test_all_df = pd.concat(all_dfs, ignore_index=True)
test_all_df = test_all_df.drop_duplicates()
test_all_df.to_csv("test_all_df.txt", sep='\t', index=False)


def load_malicious_nodes(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def load_predictions(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保不是空行
                predictions.append(int(line))
    return predictions

def load_malicious_nodes(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def load_predictions(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保不是空行
                predictions.append(int(line))
    return predictions

def test_community_zero(communities, malicious_file, prediction_file):
    malicious_nodes = load_malicious_nodes(malicious_file)
    #TODO:需要输入多个社区的预测文本，这里只用了一个测试，
    predictions = load_predictions(prediction_file)
    y_true = []
    y_pred = []
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(communities[0])):
        node_name = communities[0][i]
        pred = predictions[i]
        is_malicious = node_name in malicious_nodes
        label = 1 if is_malicious else 0
        y_true.append(label)
        y_pred.append(pred)
        if is_malicious and pred == 1:
            tp += 1
        elif not is_malicious and pred == 1:
            fp += 1
        elif is_malicious and pred == 0:
            fn += 1
        elif not is_malicious and pred == 0:
            tn += 1
    # === 计算评估指标 ===
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred)
    fpr = fp / (fp + tn + 1e-10)  # 加上一个小 epsilon 防止除0

    print("\n📊 社区 0 的评估结果：")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ AUC:       {auc:.4f}")
    print(f"✅ FPR:       {fpr:.4f}")
    print(f"✅ TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def main():
    # 👇 替换为你的真实文件路径
    malicious_path = r"D:\数据集分析\Flash-IDS-main\MaliciousLabel\M1-CVE-2015-5122_windows_h1_malicious_labels.txt"
    prediction_path = r"D:\数据集分析\Flash-IDS-main\PredictionsAnswer\M1-CVE-2015-5122_windows_h1_prediction.txt"
    result = test_community_zero(communities, malicious_path, prediction_path)

if __name__ == "__main__":
    features, edges, mapp, relations, G = prepare_graph_new(test_all_df, all_netobj2pro, all_subject2pro, all_file2pro)
    communities = detect_communities(G)
    num_nodes=len(communities[0])
    #print("社区 0 中的节点数量为:", num_nodes)
    main()



