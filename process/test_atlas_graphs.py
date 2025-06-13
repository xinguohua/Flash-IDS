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
import re
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

def evaluate_communities(communities, malicious_file, prediction_file):
    malicious_nodes = load_malicious_nodes(malicious_file)
    predictions = load_predictions(prediction_file)

    y_true = []
    y_pred = []

    tp = fp = tn = fn = 0

    for i in range(len(communities)):
        # === 获取该社区的真实标签 ===
        is_true_malicious = any(node in malicious_nodes for node in communities[i])
        y_true.append(1 if is_true_malicious else 0)
        # === 获取该社区的预测标签 ===
        # 注意：prediction_file 中的顺序应与 communities 中节点顺序一致
        is_pred_malicious = any(predictions[i] == 1 for i, node in enumerate(communities[i]))
        y_pred.append(1 if is_pred_malicious else 0)

        if is_true_malicious and is_pred_malicious:
            tp += 1
        elif not is_true_malicious and is_pred_malicious:
            fp += 1
        elif is_true_malicious and not is_pred_malicious:
            fn += 1
        elif not is_true_malicious and not is_pred_malicious:
            tn += 1

    # === 计算社区级评估指标 ===
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float('nan')
    fpr = fp / (fp + tn + 1e-10)

    print("\n📊 社区级别的评估结果：")
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
    # ⚠️ 确保 communities 已在主程序中定义（在 main 函数外的全局变量中）
    if 0 not in communities:
        print("错误：communities 中不包含编号为 0 的社区。")
        return

    result = evaluate_communities(communities, malicious_path, prediction_path)

if __name__ == "__main__":
    features, edges, mapp, relations, G = prepare_graph_new(test_all_df, all_netobj2pro, all_subject2pro, all_file2pro)
    communities = detect_communities(G)
    num_nodes=len(communities[0])
    #print("社区 0 中的节点数量为:", num_nodes)
    main()



