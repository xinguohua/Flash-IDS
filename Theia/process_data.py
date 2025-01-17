import warnings
import torch
import gdown
import os
import re

def extract_uuid(line):
    pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
    return pattern_uuid.findall(line)

def extract_subject_type(line):
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    return pattern_type.findall(line)

def show(file_path):
    print(f"Processing {file_path}")

def extract_edge_info(line):
    pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    pattern_time = re.compile(r'timestampNanos\":(.*?),')

    edge_type = extract_subject_type(line)[0]
    timestamp = pattern_time.findall(line)[0]
    src_id = pattern_src.findall(line)

    if len(src_id) == 0:
        return None, None, None, None, None

    src_id = src_id[0]
    dst_id1 = pattern_dst1.findall(line)
    dst_id2 = pattern_dst2.findall(line)

    if len(dst_id1) > 0 and dst_id1[0] != 'null':
        dst_id1 = dst_id1[0]
    else:
        dst_id1 = None

    if len(dst_id2) > 0 and dst_id2[0] != 'null':
        dst_id2 = dst_id2[0]
    else:
        dst_id2 = None

    return src_id, edge_type, timestamp, dst_id1, dst_id2


def process_data(file_path):
    id_nodetype_map = {}
    notice_num = 1000000
    for i in range(100):
        now_path = file_path + '.' + str(i)
        if i == 0:
            now_path = file_path
        if not os.path.exists(now_path):
            break

        with open(now_path, 'r') as f:
            show(now_path)
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt)

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                    continue

                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                    continue

                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                    continue

                uuid = extract_uuid(line)[0]
                subject_type = extract_subject_type(line)

                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        id_nodetype_map[uuid] = 'MemoryObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        id_nodetype_map[uuid] = 'NetFlowObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        id_nodetype_map[uuid] = 'UnnamedPipeObject'
                        continue

                id_nodetype_map[uuid] = subject_type[0]

    return id_nodetype_map


def process_edges(file_path, id_nodetype_map):
    notice_num = 1000000
    not_in_cnt = 0

    for i in range(100):
        now_path = file_path + '.' + str(i)
        if i == 0:
            now_path = file_path
        if not os.path.exists(now_path):
            break

        with open(now_path, 'r') as f, open(now_path + '.txt', 'w') as fw:
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt)

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    src_id, edge_type, timestamp, dst_id1, dst_id2 = extract_edge_info(line)

                    if src_id is None or src_id not in id_nodetype_map:
                        not_in_cnt += 1
                        continue

                    src_type = id_nodetype_map[src_id]

                    if dst_id1 is not None and dst_id1 in id_nodetype_map:
                        dst_type1 = id_nodetype_map[dst_id1]
                        this_edge1 = f"{src_id}\t{src_type}\t{dst_id1}\t{dst_type1}\t{edge_type}\t{timestamp}\n"
                        fw.write(this_edge1)

                    if dst_id2 is not None and dst_id2 in id_nodetype_map:
                        dst_type2 = id_nodetype_map[dst_id2]
                        this_edge2 = f"{src_id}\t{src_type}\t{dst_id2}\t{dst_type2}\t{edge_type}\t{timestamp}\n"
                        fw.write(this_edge2)


def run_data_processing():
    os.system('tar -zxvf ta1-theia-e3-official-1r.json.tar.gz')
    os.system('tar -zxvf ta1-theia-e3-official-6r.json.tar.gz')

    path_list = ['ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json']

    for path in path_list:
        id_nodetype_map = process_data(path)
        process_edges(path, id_nodetype_map)

    os.system('cp ta1-theia-e3-official-1r.json.txt theia_train.txt')
    os.system('cp ta1-theia-e3-official-6r.json.8.txt theia_test.txt')

# =================下载数据=========================
urls = ["https://drive.google.com/file/d/10cecNtR3VsHfV0N-gNEeoVeB89kCnse5/view?usp=drive_link",
        "https://drive.google.com/file/d/1Kadc6CUTb4opVSDE4x6RFFnEy0P1cRp0/view?usp=drive_link"]
for url in urls:
    gdown.download(url, quiet=False, use_cookies=False, fuzzy=True)

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# matplotlib inline
# =================处理数据 边/点=========================
run_data_processing()