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


def save_communities_to_txt(communities, filename="communities_atlas.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for community_id, nodes in communities.items():
            line = f"Community {community_id}: {', '.join(nodes)}\n"
            f.write(line)

    # 获取文件大小（单位：字节）
    dot_size_bytes = os.path.getsize(dot_file)
    communities_size_bytes = os.path.getsize(filename)

    dot_size_kb = dot_size_bytes / 1024
    dot_size_mb = dot_size_kb / 1024

    communities_size_kb = communities_size_bytes / 1024
    communities_size_mb = communities_size_kb / 1024
    print("压缩前:")
    print(f"文件 '{dot_file}' 大小为：{dot_size_bytes} 字节")
    print(f"约为：{dot_size_kb:.2f} KB  {dot_size_mb:.2f} MB")
    print("压缩后:")
    print(f"文件 '{filename}' 大小为：{communities_size_bytes} 字节")
    print(f"约为：{communities_size_kb:.2f} KB  {communities_size_mb:.2f} MB")

    return dot_size_bytes,communities_size_bytes

if __name__ == "__main__":
    features, edges, mapp, relations, G = prepare_graph_new(test_all_df, all_netobj2pro, all_subject2pro, all_file2pro)
    communities = detect_communities(G)
    dot_size_bytes,communities_size_bytes = save_communities_to_txt(communities)


