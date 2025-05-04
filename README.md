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



┌───────────────────┐
│   原始数据（CSV）   │
│  list of {text,label} │
└─────────┬─────────┘
          │ load_dataset
          ▼
┌───────────────────┐
│ MetaFewShotDataset│
│  • 按 label 分桶    │
│  • 随机采样 N-way K-shot + Q-query │
│  • 输出一个 episode dict：         │
│    support_input, support_mask      │
│    query_input,   query_mask        │
│    query_labels, N, K, Q            │
└─────────┬─────────┘
          │ collate_fn（batch_size>1时拼接多个 episode）
          ▼
┌───────────────────┐
│    DataLoader     │
│  （yield batch）  │
└─────────┬─────────┘
          │ Trainer.train_step 调用
          ▼
┌───────────────────┐
│   PaATModel.forward  │
│ ─────────────────── │
│ 1. Backbone 编码：   │
│    sup_seq = BERT(support_input)  │
│    qry_seq = BERT(query_input)    │
│                     │
│ 2. Dropout 分支：    │
│    sup1=sup_seq, sup2=dropout(sup_seq) │
│    qry1=qry_seq, qry2=dropout(qry_seq) │
│                     │
│ 3. 原型生成（make_protos）：       │
│    sup1 → FIAT → Pa → Adaptive → proto1 [N,H] │
│    sup2 → FIAT → Pa → Adaptive → proto2 [N,H] │
│                     │
│ 4. 查询向量（make_queries）：     │
│    qry1 → FIAT → Adaptive → xq1 [N*Q,H]       │
│    qry2 → FIAT → Adaptive → xq2 [N*Q,H]       │
│                     │
│ 5. 相似度计算：      │
│    logits = cosine(q1, p1) / temp → [N*Q, N] │
│                     │
│ 6. 返回：logits, proto1/2, xq1/2  │
└─────────┬─────────┘
          │ Trainer.compute_loss 调用
          ▼
┌───────────────────┐
│ MetaTrainer.compute_loss │
│ ─────────────────────── │
│ 1. 取出 query_labels, N, K, Q │
│ 2. 调用 model.forward → 得到 logits, proto1/2, xq1/2│
│ 3. 计算损失：            │
│    • CE = CrossEntropy(logits, query_labels) │
│    • UCL1 = 1 – cos(proto1,proto2) 平均        │
│    • UCL2 = 1 – cos(xq1, xq2) 平均             │
│    • total_loss = CE + UCL1 + UCL2            │
│ 4. 返回 loss             │
└─────────┬─────────┘
          │ 反向传播，优化器.step()
          ▼
┌───────────────────┐
│   模型参数更新     │
├───────────────────┤
│ 重复上述步骤，直到收敛 │
└───────────────────┘
