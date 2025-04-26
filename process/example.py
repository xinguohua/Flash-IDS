import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np


# from pykeen.datasets import Nations
#
# # 加载 Nations 数据集
# dataset = Nations()
#
# # 获取训练数据的 triples_factory
# triples_factory = dataset.training
#
# # 打印前 5 条三元组数据
# print("Sample triples (head, relation, tail):")
# print(triples_factory.triples[:5])
#
# # 打印所有实体
# print("\nEntities:")
# print(triples_factory.entity_to_id)
#
# # 打印所有关系
# print("\nRelations:")
# print(triples_factory.relation_to_id)
#
# # 运行 TransE 模型
# result = pipeline(
#     model='TransE',
#     dataset='nations',  # 选择一个内置的示例数据集
#     model_kwargs={'embedding_dim': 5},
#     training_kwargs={'num_epochs': 100, 'batch_size': 256},
# )
#
# # 获取训练好的模型
# trained_model = result.model
#
# # 使用 `result.training` 获取 `triples_factory`
# triples_factory = result.training
#
# # 选择一个实体并获取其嵌入向量
# entity_id = triples_factory.entity_to_id['brazil']
# entity_tensor = torch.tensor([entity_id], dtype=torch.long)
# print(trained_model.entity_representations)
# entity_embedding = trained_model.entity_representations[0](
#     indices=entity_tensor
# ).detach().cpu().numpy()
#
# print(f"Brazil entity embedding: {entity_embedding}")
#
# # 获取关系的嵌入向量
# relation_to_id = triples_factory.relation_to_id  # 获取关系索引
# relation_name = "aidenemy"  # 你要查询的关系名称
# relation_id = relation_to_id[relation_name]  # 关系的索引
# relation_tensor = torch.tensor([relation_id], dtype=torch.long)  # 转成 tensor
# relation_embedding = trained_model.relation_representations[0](
#     indices=relation_tensor
# ).detach().cpu().numpy()
#
# print(f"Relation '{relation_name}' embedding: {relation_embedding}")

# 手动创建数据集**
# 定义简单的三元组 (头实体, 关系, 尾实体)
triples = np.array([
    ["brazil", "exports", "usa"],
    ["usa", "trades_with", "germany"],
    ["germany", "aidenemy", "france"],
    ["france", "supports", "brazil"],
    ["brazil", "aidenemy", "germany"],
], dtype=object)

# ** 创建 TriplesFactory**
triples_factory = TriplesFactory.from_labeled_triples(triples)
training, validation, testing = triples_factory.split([0.8, 0.1, 0.1])

# ** 运行 TransE 模型**
result = pipeline(
    model='TransE',
    training=triples_factory,
    validation=validation,
    testing=testing,
    model_kwargs={'embedding_dim': 5},
    training_kwargs={'num_epochs': 100, 'batch_size': 16},
)

# **
trained_model = result.model

# **获取 "brazil" 的实体嵌入**
entity_to_id = triples_factory.entity_to_id  # 获取实体索引
entity_id = entity_to_id["brazil"]
entity_tensor = torch.tensor([entity_id], dtype=torch.long)

# 打印实体嵌入
print(trained_model.entity_representations)
entity_embedding = trained_model.entity_representations[0](
    indices=entity_tensor
).detach().cpu().numpy()
print(f"Brazil entity embedding: {entity_embedding}")

# ** 获取 "aidenemy" 关系的嵌入**
relation_to_id = triples_factory.relation_to_id  # 获取关系索引
relation_id = relation_to_id["aidenemy"]  # 关系的索引
relation_tensor = torch.tensor([relation_id], dtype=torch.long)  # 转成 tensor

relation_embedding = trained_model.relation_representations[0](
    indices=relation_tensor
).detach().cpu().numpy()
print(f"Relation 'aidenemy' embedding: {relation_embedding}")
