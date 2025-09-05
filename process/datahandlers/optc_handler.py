import json
import os
import re
import igraph as ig
import pandas as pd
from process.optc_type_enum import optcObjectType
from .base import BaseProcessor
from .common import (
    collect_json_paths, collect_label_paths,
    merge_properties, add_node_properties, get_or_add_node, add_edge_if_new,
    update_edge_index
)

class OptcHandler(BaseProcessor):
    def load(self):
        json_map = collect_json_paths(self.base_path)
        label_map = collect_label_paths(self.base_path)

        for scene, category_data in json_map.items():
            # 只处理 day1
            if scene != "0402":
                continue

            # 处理 label
            if not self.train:
                with open(label_map[scene], "r") as label_file:
                    print(f"[INFO] 正在处理: 场景={scene}, label={label_map[scene]}")
                    labels = [line.strip() for line in label_file.read().splitlines() if line.strip()]
                    print(f"[INFO] 加载到 {len(labels)} 条标签")
                    self.all_labels.extend(labels)

            for category, json_files in category_data.items():
                # 训练只要 benign
                if self.train and category != "benign":
                    continue
                # 测试只要 benign + malicious
                if not self.train and category not in ["benign", "malicious"]:
                    continue

                for jf in json_files:  # 遍历每一个 JSON 文件
                    abs_json_path = os.path.abspath(jf)  # 确保是绝对路径
                    print(f"\n[STEP] 正在处理: 场景={scene}, 类别={category}, JSON 文件={abs_json_path}")

                    if not os.path.isfile(abs_json_path):
                        print(f"[WARN] JSON 文件不存在: {abs_json_path}, 跳过")
                        continue

                    # 对应的 TXT 文件
                    dir_name = os.path.dirname(jf)
                    # 'AIA-401-425.json'
                    base_name = os.path.basename(jf)
                    name, _ext = os.path.splitext(base_name)
                    # 要拼接成
                    # ../../data_files_optc/day1/0402_benign_AIA-401-425.txt
                    # 拆分目录
                    parent_dir = os.path.dirname(os.path.dirname(dir_name))  # ../../data_files_optc/day1
                    last1 = os.path.basename(os.path.dirname(dir_name))  # 0402
                    last2 = os.path.basename(dir_name)  # benign
                    prefix = f"{last1}_{last2}"  # 0402_benign
                    txt_path = os.path.join(parent_dir, f"{prefix}_{name}.txt")
                    if not os.path.isfile(txt_path):
                        print(f"[WARN] 找不到对应 TXT 文件: {txt_path}, 跳过")
                        continue

                    # 读取 TXT -> df
                    df = _read_optc_txt_as_df(txt_path)
                    print(f"[INFO] 读取 {txt_path}, 初始 df 行数={len(df)}")

                    df = df.dropna()
                    df.sort_values(by="timestamp", ascending=True, inplace=True)
                    print(f"[INFO] 清理后 df 行数={len(df)}")

                    # 节点和边收集（使用绝对路径 JSON）
                    netobj2pro, subject2pro, file2pro = collect_nodes_from_log_optc([abs_json_path])
                    df = collect_edges_from_log_optc(df, [abs_json_path])
                    print(f"[INFO] 收集到: 网络对象={len(netobj2pro)}, 主体={len(subject2pro)}, 文件={len(file2pro)}")
                    print(f"[INFO] 添加边后 df 行数={len(df)}")

                    # 切分数据
                    if self.train:
                        cut_idx = int(len(df) * 0.9)
                        df = df.iloc[:self.max_benign_lines]
                        print(f"[INFO] 训练模式 -> 保留前 90%, 行数={len(df)}")
                    else:
                        if category == "benign":
                            cut_idx = int(len(df) * 0.9)
                            df = df.iloc[cut_idx:]
                            print(f"[INFO] 测试模式 benign -> 保留后 10%, 行数={len(df)}")
                        else:
                            print(f"[INFO] 测试模式 malicious -> 使用全部, 行数={len(df)}")

                    # 存储
                    if not df.empty:
                        self.all_dfs.append(df)
                        print(f"[SAVE] 已加入 self.all_dfs, 当前累计 {len(self.all_dfs)} 个 df")
                    else:
                        print(f"[WARN] df 为空, 跳过存储")

                    merge_properties(netobj2pro, self.all_netobj2pro)
                    merge_properties(subject2pro, self.all_subject2pro)
                    merge_properties(file2pro, self.all_file2pro)

        # 合并
        if self.all_dfs:
            self.use_df = pd.concat(self.all_dfs, ignore_index=True).drop_duplicates()
            print(f"\n[FINAL] 合并完成, self.use_df 行数={len(self.use_df)}")
        else:
            self.use_df = pd.DataFrame(columns=[
                'actorID', 'actor_type', 'objectID', 'object',
                'action', 'timestamp', 'exec', 'path'
            ])
            print(f"\n[FINAL] 没有生成任何 df, 返回空 DataFrame")

    def build_graph(self):
        G = ig.Graph(directed=True)
        nodes, edges, relations = {}, [], {}

        for _, row in self.use_df.iterrows():
            action = row.get("action", "")

            actor_id = row.get("actorID", "")
            object_id = row.get("objectID", "")

            # 节点属性（字符串特征）
            properties_actor = extract_properties_optc(
                actor_id, row, action,
                self.all_netobj2pro, self.all_subject2pro, self.all_file2pro
            )
            properties_object = extract_properties_optc(
                object_id, row, action,
                self.all_netobj2pro, self.all_subject2pro, self.all_file2pro
            )

            add_node_properties(nodes, actor_id, properties_actor)
            add_node_properties(nodes, object_id, properties_object)

            edge = (actor_id, object_id)
            edges.append(edge)
            relations[edge] = action

            # ---- 构图（节点类型用枚举） ----
            actor_idx = get_or_add_node(G, actor_id, optcObjectType[row['actor_type']].value, properties_actor)
            object_idx = get_or_add_node(G, object_id, optcObjectType[row['object']].value, properties_object)

            # 节点标签
            G.vs[actor_idx]["label"] = int(actor_id in self.all_labels)
            G.vs[object_idx]["label"] = int(object_id in self.all_labels)

            # 添加边
            add_edge_if_new(G, actor_idx, object_idx, action)

        # ---- 构建 features 和 edge_index ----
        features, edge_index, index_map, relations_index = [], [[], []], {}, {}
        for node_id, props in nodes.items():
            features.append(props)
            index_map[node_id] = len(features) - 1

        update_edge_index(edges, edge_index, index_map, relations, relations_index)

        return features, edge_index, list(index_map.keys()), relations_index, G


def _read_optc_txt_as_df(txt_path):
    df = pd.read_csv(txt_path, sep=r"\t| {2,}|\s{1}", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {"Source_ID":"actorID","Source_Type":"actor_type","Destination_ID":"objectID",
                  "Destination_Type":"object","Edge_Type":"action","Timestamp":"timestamp"}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    for c in ["actorID","actor_type","objectID","object","action","timestamp"]:
        df[c] = df[c].astype(str)
    return df[["actorID","actor_type","objectID","object","action","timestamp"]]


def iter_json_records(json_path):
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read().strip()
    if not data:
        return []
    try:
        arr = json.loads(data)
        if isinstance(arr,list):
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
            return
    except:
        pass
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks = re.split(r"}\s*{\s*", line)
        if len(chunks) > 1:
            chunks[0] += "}"
            chunks[-1] = "{" + chunks[-1]
            for c in chunks:
                try:
                    yield json.loads(c)
                except: continue
        else:
            try: yield json.loads(line)
            except: continue


def collect_nodes_from_log_optc(paths):
    netobj2pro, subject2pro, file2pro = {}, {}, {}
    for p in paths:
        for rec in iter_json_records(p):
            obj_type = str(rec.get("object","")).upper()
            obj_id = str(rec.get("objectID",""))
            props = rec.get("properties",{}) or {}

            if obj_type=="FILE":
                file2pro[obj_id] = props.get("file_path","")
            elif obj_type=="PROCESS":
                node_property = ",".join([
                    props.get("command_line", ""),
                    str(rec.get("tgid","")),
                    props.get("image_path", "")
                ])
                subject2pro[obj_id] = node_property
            elif obj_type in ["FLOW","NETFLOW"]:
                node_property = ",".join([
                    props.get("src_ip",""), props.get("src_port",""),
                    props.get("dest_ip",""), props.get("dest_port","")
                ])
                netobj2pro[obj_id] = node_property
    return netobj2pro, subject2pro, file2pro


def collect_edges_from_log_optc(d, paths):
    info = []
    for p in paths:
        for x in iter_json_records(p):
            action = str(x.get("action",""))
            actor = str(x.get("actorID",""))
            obj = str(x.get("objectID",""))
            ts = str(x.get("timestamp",""))
            props = x.get("properties",{}) or {}
            cmd = str(props.get("command_line","") or "")
            path = str(props.get("image_path","") or "")
            info.append({'actorID':actor,'objectID':obj,'action':action,'timestamp':ts,'exec':cmd,'path':path})
    rdf = pd.DataFrame.from_records(info).astype(str)
    return d.merge(rdf, how='inner', on=['actorID','objectID','action','timestamp']).drop_duplicates()


def extract_properties_optc(node_id, row, action, netobj2pro, subject2pro, file2pro):
    if node_id in netobj2pro: return netobj2pro[node_id]
    if node_id in file2pro: return file2pro[node_id]
    if node_id in subject2pro: return subject2pro[node_id]
    parts = [str(row.get("exec","")), action, str(row.get("path",""))]
    return " ".join([p for p in parts if p])