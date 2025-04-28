from transformers import TrainingArguments
from model.smtp_bert_re import SMTPBert
from dataset.smtp_dataset_loader_re import loader_dataset
from trainer.smtp_trainer_re import CustomTrainer, CustomCallback


# 主函数
if __name__ == "__main__":

    tokenizer_name = "bert-base-uncased"
    model_name = "bert-base-uncased"
    datasetname = ["banking77_csv"]
    # 加载数据集
    merged_dataset = loader_dataset(datasetname, tokenizer_name)

    # 初始化模型
    model = SMTPBert(model_name)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="module",  # 输出目录，用于保存模型检查点和日志
        num_train_epochs=3,  # 训练的总 epoch 数
        per_device_train_batch_size=32,  # 每个设备的训练批次大小
        save_strategy="epoch",  # 每个 epoch 结束时保存模型检查点
        learning_rate=2e-5,  # 学习率
        weight_decay=0.01,  # 权重衰减系数
        warmup_steps=100,  # 学习率预热步数
        logging_dir="logs",  # 日志目录
        logging_steps=10,  # 每隔 10 步记录日志
        save_total_limit=2,  # 最多保存 2 个模型检查点
        fp16=False,  # 使用混合精度训练（如果设备GPU支持）
        push_to_hub=False,  # 不将模型推送到 Hugging Face Model Hub
        report_to="tensorboard",  # 报告训练进度到 TensorBoard
    )

    # 初始化 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=merged_dataset,
        callbacks=[
            CustomCallback(),
        ],
    )
    # 开始训练
    trainer.train()
    # 保存模型
    model.save_pretrained(training_args.output_dir)
