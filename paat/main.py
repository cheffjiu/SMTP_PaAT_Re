from transformers import TrainingArguments, AutoTokenizer
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
# 修改此处，不再只取训练集
# raw_dataset = raw_dataset["train"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_dataset = MetaFewShotDataset(
    data=raw_dataset["train"],
    tokenizer=tokenizer,
    N=5,  # 5-way
    K=1,  # 1-shot
    Q=5,  # 每类 5 个 query
    max_length=64,
)

val_dataset = MetaFewShotDataset(
    data=raw_dataset["val"],
    tokenizer=tokenizer,
    N=5,  # 5-way
    K=1,  # 1-shot
    Q=5,  # 每类 5 个 query
    max_length=64,
)

test_dataset = MetaFewShotDataset(
    data=raw_dataset["test"],
    tokenizer=tokenizer,
    N=5,  # 5-way
    K=1,  # 1-shot
    Q=5,  # 每类 5 个 query
    max_length=64,
)

# 2. 模型
model = PaATModel("bert-base-uncased", temp=0.5)

# 3. 训练参数
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,  # 一个 episode 作为一个 batch
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    logging_dir="./logs",
    #eval_strategy="epoch",  # 添加验证策略
    #save_strategy="epoch",  # 添加保存策略，与 eval_strategy 保持一致
    #eval_steps=10,  # 每 50 步进行一次验证
    #load_best_model_at_end=True,  # 训练结束后加载最佳模型
)

# 4. Trainer
trainer = MetaTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=val_dataset,  # 添加验证集
   
)

# 5. 启动训练
trainer.train()

# 测试模型
results = trainer.evaluate(test_dataset)
print("Test results:", results)
