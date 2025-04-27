import torch
from transformers import Trainer


# 自定义 Trainer 用于计算损失
class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 打印模型所在设备
        print(f"Model device: {next(model.parameters()).device}")
        # 打印输入数据所在设备
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} device: {value.device}")

        input_ids_mlm = inputs.pop("input_ids_mlm")
        attention_mask_mlm = inputs.pop("attention_mask_mlm")
        labels_mlm = inputs.pop("labels_mlm")
        input_ids_contrast1 = inputs.pop("input_ids_contrast1")
        attention_mask_contrast1 = inputs.pop("attention_mask_contrast1")
        input_ids_contrast2 = inputs.pop("input_ids_contrast2")
        attention_mask_contrast2 = inputs.pop("attention_mask_contrast2")
        input_ids_contrast3 = inputs.pop("input_ids_contrast3")
        attention_mask_contrast3 = inputs.pop("attention_mask_contrast3")

        loss = model(
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
        return (loss, None) if return_outputs else loss
