from transformers import TrainingArguments, AutoTokenizer, Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset.paat_dataset_loader_re import MetaFewShotDataset
from model.paat_re import PaATModel
from trainer.paat_trainer_re import MetaTrainer


# 0) 加载数据集
data_files = {
    "train": "data/banking77_meta_csv/meta_train.csv",
    "test": "data/banking77_meta_csv/meta_test.csv",
    "val": "data/banking77_meta_csv/meta_val.csv",
}
raw_dataset = load_dataset("csv", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

N = 5
K = 1
Q = 5

train_dataset = MetaFewShotDataset(
    data=raw_dataset["train"],
    tokenizer=tokenizer,
    N=N,  # 5-way
    K=K,  # 1-shot
    Q=Q,  # 每类 5 个 query
    max_length=64,
)

val_dataset = MetaFewShotDataset(
    data=raw_dataset["val"],
    tokenizer=tokenizer,
    N=N,  # 5-way
    K=K,  # 1-shot
    Q=Q,  # 每类 5 个 query
    max_length=64,
)

test_dataset = MetaFewShotDataset(
    data=raw_dataset["test"],
    tokenizer=tokenizer,
    N=N,  # 5-way
    K=K,  # 1-shot
    Q=Q,  # 每类 5 个 query
    max_length=64,
)

# 2. 模型
model = PaATModel("bert-base-uncased", N, K, Q, temp=0.5)


# 3. 训练参数
training_args = TrainingArguments(
    output_dir="outputs",
    logging_dir="logs",
    report_to=["tensorboard"],
    eval_strategy="steps",          # 保持步数评估策略
    eval_steps=5,                   # 每5个训练步做一次验证
    logging_strategy="steps",
    logging_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=20,           # 增加最大epoch数
    load_best_model_at_end=True,    # 必须设置为True
    metric_for_best_model="eval_f1_macro",  # 根据验证集f1_macro选择最佳模型
    greater_is_better=True,
    learning_rate=2e-5,             # 增加学习率
    save_strategy="steps",          # 保存策略与评估策略对齐
    save_steps=5,                   # 保存间隔与评估间隔一致
    save_total_limit=3,             # 限制checkpoint数量
    gradient_accumulation_steps=1,  # 明确梯度累积步数
    warmup_steps=100,               # 添加学习率预热
)
# 4. Trainer
trainer = MetaTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()


# 5. 启动训练
trainer.train()

# 测试模型
results = trainer.evaluate(test_dataset)
print("Test results:", results)
