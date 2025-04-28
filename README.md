# SMTP_PaAT_Re
复现Few-shot intent detection with self-supervised pretraining and prototype-aware attention
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",  # 输出目录，用于保存模型检查点和日志
    num_train_epochs=3,  # 训练的总 epoch 数
    per_device_train_batch_size=32,  # 每个设备的训练批次大小
    per_device_eval_batch_size=32,  # 每个设备的评估批次大小
    evaluation_strategy="epoch",  # 每个 epoch 结束时进行评估
    save_strategy="epoch",  # 每个 epoch 结束时保存模型检查点
    learning_rate=2e-5,  # 初始学习率
    weight_decay=0.01,  # 权重衰减系数
    warmup_steps=100,  # 学习率预热步数
    logging_dir="./logs",  # 日志目录
    logging_steps=10,  # 每隔 10 步记录日志
    save_total_limit=2,  # 最多保存 2 个模型检查点
    fp16=True,  # 使用混合精度训练（如果设备支持）
    load_best_model_at_end=True,  # 训练结束时加载最佳模型
    metric_for_best_model="loss",  # 用于选择最佳模型的指标（例如 loss 或 accuracy）
    greater_is_better=False,  # 指标是否越大越好（对于 loss 通常为 False）
    push_to_hub=False,  # 不将模型推送到 Hugging Face Model Hub
)