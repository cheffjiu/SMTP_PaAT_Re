from transformers import Trainer, TrainerCallback
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup
import torch
import torch.nn as nn


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 检测当前使用的设备
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        # 对比学习损失函数
        self.cl_head_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        重写 compute_loss 方法，计算自定义损失。

        Args:
            model: 自定义模型实例。
            inputs: 输入数据字典。
            return_outputs (bool, optional): 是否返回模型输出。默认为 False。
            num_items_in_batch: 批次中的样本数量，可选参数。

        Returns:
            若 return_outputs 为 True,返回 (loss, outputs) 元组；否则返回 loss。
        """
        input_ids_mlm = inputs.pop("input_ids_mlm")
        attention_mask_mlm = inputs.pop("attention_mask_mlm")
        labels_mlm = inputs.pop("labels_mlm")
        input_ids_contrast1 = inputs.pop("input_ids_contrast1")
        attention_mask_contrast1 = inputs.pop("attention_mask_contrast1")
        input_ids_contrast2 = inputs.pop("input_ids_contrast2")
        attention_mask_contrast2 = inputs.pop("attention_mask_contrast2")
        input_ids_contrast3 = inputs.pop("input_ids_contrast3")
        attention_mask_contrast3 = inputs.pop("attention_mask_contrast3")

        mlm_outputs, embeddings1, embeddings2, embeddings3 = model(
            input_ids_mlm,
            attention_mask_mlm,
            labels_mlm,
            input_ids_contrast1,
            attention_mask_contrast1,
            input_ids_contrast2,
            attention_mask_contrast2,
            input_ids_contrast3,
            attention_mask_contrast3,
        )
        mlm_loss = mlm_outputs.loss
        contrast_loss = self.cl_head_loss(embeddings1, embeddings2, embeddings3)
        # 计算总损失
        total_loss = mlm_loss + contrast_loss
        return (
            (total_loss, (mlm_outputs, embeddings1, embeddings2, embeddings3))
            if return_outputs
            else total_loss
        )

    def create_optimizer(self):
        """
        重写 create_optimizer 方法，使用 Adam 优化器。
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = optim.Adam(
                optimizer_grouped_parameters, lr=self.args.learning_rate
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        重写 create_scheduler 方法，使用线性学习率调度器。
        """
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 在日志记录事件中，将训练指标数据保存到文件
        with open("logs/training_logs.txt", "a") as f:
            f.write(
                f"Epoch: {state.epoch}, Step: {state.global_step}, Loss: {logs.get('loss', 'N/A')}\n"
            )
