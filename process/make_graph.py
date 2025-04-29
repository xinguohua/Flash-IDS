import pandas as pd
import json
import igraph as ig
from type_enum import ObjectType

# =================处理特征成图=========================
def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)


def update_edge_index(edges, edge_index, index, relations, relations_index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

        relation = relations[(src_id, dst_id)]
        relations_index[(src, dst)] = relation

# TODO：特征要补充啊
# 成图+捕捉特征语料+简化策略这里添加
def prepare_graph(df):
    G = ig.Graph(directed=True)
    nodes, labels, edges, relations = {}, {}, [], {}
    # dummies = {"SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_BLOCK": 2,
    #            "NetFlowObject": 3, "PRINCIPAL_REMOTE": 4, 'PRINCIPAL_LOCAL': 5}

    for _, row in df.iterrows():
        action = row["action"]
        properties = [row['exec'], action] + ([row['path']] if row['path'] else [])

        actor_id = row["actorID"]
        add_node_properties(nodes, actor_id, properties)
        labels[actor_id] = ObjectType[row['actor_type']].value

        object_id = row["objectID"]
        add_node_properties(nodes, object_id, properties)
        labels[object_id] = ObjectType[row['object']].value

        edge = (actor_id, object_id)
        edges.append(edge)
        relations[edge] = action

        # 初始化igraph的图
        G.add_vertices(1)
        G.vs[len(G.vs)-1]['name'] = actor_id
        G.vs[len(G.vs)-1]['type'] = ObjectType[row['actor_type']].value
        G.vs[len(G.vs)-1]['properties'] = properties
        G.add_vertices(1)
        G.vs[len(G.vs)-1]['name'] = object_id
        G.vs[len(G.vs)-1]['type'] = ObjectType[row['object']].value
        G.vs[len(G.vs)-1]['properties'] = properties
        G.add_edges([(actor_id, object_id)])
        G.es[len(G.es)-1]['actions'] = action

    features, feat_labels, edge_index, index_map, relations_index = [], [], [[], []], {}, {}
    for node_id, props in nodes.items():
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1

    update_edge_index(edges, edge_index, index_map, relations, relations_index)

    return features, feat_labels, edge_index, list(index_map.keys()), relations_index, G

# TODO：特征要补充啊
# 成图+捕捉特征语料+简化策略这里添加
def prepare_graph_new(df):
    G = ig.Graph(directed=True)
    nodes, edges, relations = {}, [], {}

    for _, row in df.iterrows():
        action = row["action"]
        properties = [row['exec'], action] + ([row['path']] if row['path'] else [])

        actor_id = row["actorID"]
        add_node_properties(nodes, actor_id, properties)

        object_id = row["objectID"]
        add_node_properties(nodes, object_id, properties)

        edge = (actor_id, object_id)
        edges.append(edge)
        relations[edge] = action

        # 初始化igraph的图
        G.add_vertices(1)
        G.vs[len(G.vs) - 1]['name'] = actor_id
        G.vs[len(G.vs) - 1]['type'] = ObjectType[row['actor_type']].value
        G.vs[len(G.vs) - 1]['properties'] = properties
        G.add_vertices(1)
        G.vs[len(G.vs) - 1]['name'] = object_id
        G.vs[len(G.vs) - 1]['type'] = ObjectType[row['object']].value
        G.vs[len(G.vs) - 1]['properties'] = properties
        G.add_edges([(actor_id, object_id)])
        G.es[len(G.es) - 1]['actions'] = action

    features, feat_labels, edge_index, index_map, relations_index = [], [], [[], []], {}, {}
    for node_id, props in nodes.items():
        features.append(props)
        index_map[node_id] = len(features) - 1

    update_edge_index(edges, edge_index, index_map, relations, relations_index)

    return features, edge_index, list(index_map.keys()), relations_index, G

def add_attributes(d, p):
    f = open(p)
    # data = [json.loads(x) for x in f if "EVENT" in x]
    # for test
    data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 300]
    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['cmdLine']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp, 'exec': cmd,
                         'path': path2})
        except:
            pass

        info.append(
            {'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path})

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


def add_attributes_new(d, paths):
    info = []
    for p in paths:
        with open(p) as f:
            # TODO
            # for test: 只取每个文件前300条包含"EVENT"的
            data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 1000]
            # data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x ]

        for x in data:
            try:
                action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
            except:
                action = ''
            try:
                actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
            except:
                actor = ''
            try:
                obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject'][
                    'com.bbn.tc.schema.avro.cdm18.UUID']
            except:
                obj = ''
            try:
                timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
            except:
                timestamp = ''
            try:
                cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['cmdLine']
            except:
                cmd = ''
            try:
                path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
            except:
                path = ''
            try:
                path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
            except:
                path2 = ''
            try:
                obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2'][
                    'com.bbn.tc.schema.avro.cdm18.UUID']
                info.append({
                    'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp,
                    'exec': cmd, 'path': path2
                })
            except:
                pass

            info.append({
                'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp,
                'exec': cmd, 'path': path
            })

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


