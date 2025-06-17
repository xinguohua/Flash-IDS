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
from datahandlers.darpa_handler import collect_nodes_from_log,collect_edges_from_log
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
all_dfs = []
all_netobj2pro = {}  # 网络对象 UUID → 属性字符串
all_subject2pro = {}  # 进程 UUID → 属性字符串
all_file2pro = {}  # 文件 UUID → 属性字符串
all_json_files = set()
def merge_properties(src_dict, target_dict):
    for k, v in src_dict.items():
        if k not in target_dict:
            target_dict[k] = v

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

for scene, category_data in json_map.items():
    for category, json_files in category_data.items():
        # TODO: for test
        if scene != "theia33":
            continue
        print(f"正在处理: 场景={scene}, 类别={category}, 文件={json_files}")
        for json_path in json_files:
            all_json_files.add(json_path)
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

        # 形成一个更完整的视图
        netobj2pro, subject2pro, file2pro = collect_nodes_from_log(json_files)
        df = collect_edges_from_log(df, json_files)
        # 数据选择逻辑
        if category == "benign":
            # 取后10%
            num_rows = int(len(df) * 0.9)
            df = df.iloc[num_rows:]
            all_dfs.append(df)
        elif category == "malicious":
            # 使用全部
            all_dfs.append(df)
            pass
        else:
            continue

        merge_properties(netobj2pro, all_netobj2pro)
        merge_properties(subject2pro, all_subject2pro)
        merge_properties(file2pro, all_file2pro)

# 测试用的数据集
test_all_df = pd.concat(all_dfs, ignore_index=True)
test_all_df = test_all_df.drop_duplicates()
test_all_df.to_csv("test_all_df.txt", sep='\t', index=False)

def save_communities_to_txt(communities, filename="communities_darpa.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for community_id, nodes in communities.items():
            line = f"Community {community_id}: {', '.join(nodes)}\n"
            f.write(line)
    total_size_bytes = 0
    for json_path in all_json_files:
        print(json_path)
        total_size_bytes += os.path.getsize(json_path)
    json_size_kb = total_size_bytes / 1024
    json_size_mb = json_size_kb / 1024
    communities_size_bytes = os.path.getsize(filename)
    communities_size_kb = communities_size_bytes / 1024
    communities_size_mb = communities_size_kb / 1024
    print("压缩前:")
    print(f"大小为：{total_size_bytes} 字节")
    print(f"约为：{json_size_kb:.2f} KB  {json_size_mb:.2f} MB")
    print("压缩后:")
    print(f"文件 '{filename}' 大小为：{communities_size_bytes} 字节")
    print(f"约为：{communities_size_kb:.2f} KB  {communities_size_mb:.2f} MB")

    return total_size_bytes,communities_size_bytes

if __name__ == "__main__":
    features, edges, mapp, relations, G = prepare_graph_new(test_all_df, all_netobj2pro, all_subject2pro, all_file2pro)
    communities = detect_communities(G)
    json_size_bytes,commnities_size_bytes = save_communities_to_txt(communities)




