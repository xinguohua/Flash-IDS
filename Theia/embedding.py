from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch
import numpy as np

# 定义自定义三元组数据
triples = np.array([
    ("ProcessA", "creates", "File1"),
    ("ProcessA", "communicates_with", "ProcessB"),
    ("ProcessB", "writes", "File2"),
    ("ProcessC", "reads", "File1"),
    ("ProcessC", "creates", "File3"),
    ("ProcessA", "creates", "File3"),
    ("ProcessB", "communicates_with", "ProcessC"),
    ("ProcessD", "creates", "File4"),
    ("ProcessE", "writes", "File5"),
    ("ProcessF", "reads", "File2"),
    ("ProcessG", "creates", "File6"),
    ("ProcessH", "writes", "File1"),
    ("ProcessI", "communicates_with", "ProcessJ"),
    ("ProcessJ", "creates", "File7"),
    ("ProcessA", "writes", "File8"),
    ("ProcessD", "reads", "File4"),
    ("ProcessE", "creates", "File5"),
    ("ProcessG", "writes", "File9"),
    ("ProcessH", "reads", "File10"),
    ("ProcessI", "creates", "File11"),
], dtype=object)


# 利用三元组数据构造 TriplesFactory
triples_factory = TriplesFactory.from_labeled_triples(triples)
training, validation, testing = triples_factory.split([0.8, 0.1, 0.1])

# 运行 pipeline，使用 TransE 模型进行训练
result = pipeline(
    model='TransE',
    training=triples_factory,
    validation=validation,
    testing=testing,
    model_kwargs={'embedding_dim': 5},
    training_kwargs={'num_epochs': 100, 'batch_size': 16},
)

trained_model = result.model
# **获取的实体嵌入**
entity_to_id = triples_factory.entity_to_id  # 获取实体索引
entity_id = entity_to_id["ProcessA"]
entity_tensor = torch.tensor([entity_id], dtype=torch.long)
entity_embedding = trained_model.entity_representations[0](
    indices=entity_tensor
).detach().cpu().numpy()
print(f"ProcessA entity embedding: {entity_embedding}")

# ** 获取关系的嵌入**
relation_to_id = triples_factory.relation_to_id  # 获取关系索引
relation_id = relation_to_id["creates"]  # 关系的索引
relation_tensor = torch.tensor([relation_id], dtype=torch.long)  # 转成 tensor
relation_embedding = trained_model.relation_representations[0](
    indices=relation_tensor
).detach().cpu().numpy()
print(f"Relation 'creates' embedding: {relation_embedding}")

