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
                        df = df.iloc[:cut_idx]
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
        use_df = self.use_df
        all_labels = set(self.all_labels)

        _otype_cache = {}

        def _otype(v):
            if v not in _otype_cache:
                _otype_cache[v] = optcObjectType[v].value
            return _otype_cache[v]

        nodes_props, nodes_type, edges_map = {}, {}, {}

        # === 扫描 DataFrame 收集节点与边 ===
        for r in use_df.itertuples(index=False):
            action = getattr(r, "action")
            actor_id = getattr(r, "actorID")
            object_id = getattr(r, "objectID")

            # actor 节点
            props_actor = extract_properties_optc(actor_id, r, action,
                                             self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes_props, actor_id, props_actor)
            if actor_id not in nodes_type:
                nodes_type[actor_id] = _otype(getattr(r, "actor_type"))

            # object 节点
            props_obj = extract_properties_optc(object_id, r, action,
                                           self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes_props, object_id, props_obj)
            if object_id not in nodes_type:
                nodes_type[object_id] = _otype(getattr(r, "object"))

            # 累加动作到 set
            edges_map.setdefault((actor_id, object_id), set()).add(action)

        # === 创建图节点 ===
        node_ids = list(nodes_props.keys())
        index_map = {nid: i for i, nid in enumerate(node_ids)}

        G = ig.Graph(directed=True)
        G.add_vertices(len(node_ids))
        G.vs["name"] = node_ids
        G.vs["type"] = [nodes_type.get(nid) for nid in node_ids]
        G.vs["properties"] = [nodes_props[nid] for nid in node_ids]
        G.vs["label"] = [1 if nid in all_labels else 0 for nid in node_ids]

        # === 创建图边 ===
        unique_edges = list(edges_map.keys())
        if unique_edges:
            edge_idx = [(index_map[a], index_map[b]) for (a, b) in unique_edges]
            G.add_edges(edge_idx)
            # 每条边的 action 是 list(set)
            G.es["action"] = [list(edges_map[(a, b)]) for (a, b) in unique_edges]

        # === 下游需要的结构 ===
        features = [nodes_props[nid] for nid in node_ids]
        edge_index = [[], []]
        relations_index = {}
        for a, b in unique_edges:
            s, d = index_map[a], index_map[b]
            edge_index[0].append(s)
            edge_index[1].append(d)
            relations_index[(s, d)] = list(edges_map[(a, b)])

        return features, edge_index, node_ids, relations_index, G


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
    if node_id in netobj2pro:
        return netobj2pro[node_id]
    elif node_id in file2pro:
        return file2pro[node_id]
    elif node_id in subject2pro:
        return subject2pro[node_id]
    else:
        exec_cmd = getattr(row, "exec", "")
        path_val = getattr(row, "path", "")
        return " ".join([exec_cmd, action] + ([path_val] if path_val else []))