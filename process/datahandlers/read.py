import pandas as pd

files = [
    "../../data_files5/cadets/cadets93_benign.txt",
    "../../data_files5/cadets/cadets93_malicious.txt",
    "../../data_files5/cadets/cadets104_benign.txt",
    "../../data_files5/cadets/cadets104_malicious.txt"
]

df_list = []
for file in files:
    with open(file, "r") as f:
        data = f.read().split('\n')
        data = [line.split('\t') for line in data]
        df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
        df = df.dropna()
        df_list.append(df)

# 合并
all_df = pd.concat(df_list, ignore_index=True)

# 去重统计
unique_objects = all_df['object'].unique()
print("所有不同的 object 类型：", unique_objects)
print("总共有", all_df['object'].nunique(), "种不同的 object 类型")