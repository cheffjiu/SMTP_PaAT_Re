import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from typing import Any, Dict, Optional


class SMTPBert(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        # 主干网络使用AutoModelForMaskedLM 模型
        self.base = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

        # 语义对比学习头
        self.cl_head = nn.Sequential(
            nn.Linear(self.base.config.hidden_size, self.base.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.base.config.hidden_dropout_prob),
            nn.Linear(self.base.config.hidden_size, 256),
        )

        # 对比学习损失函数
        self.cl_head_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(
        self,
        input_ids_mlm: torch.Tensor,
        attention_mask_mlm: torch.Tensor,
        labels_mlm: torch.Tensor,
        input_ids_contrast1: torch.Tensor,
        attention_mask_contrast1: torch.Tensor,
        input_ids_contrast2: torch.Tensor,
        attention_mask_contrast2: torch.Tensor,
        input_ids_contrast3: torch.Tensor,
        attention_mask_contrast3: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播方法。

        参数:
        input_ids_mlm (torch.Tensor): 掩码语言模型的输入 ID。
        attention_mask_mlm (torch.Tensor): 掩码语言模型的注意力掩码。
        labels_mlm (torch.Tensor): 掩码语言模型的标签。
        input_ids_contrast1 (torch.Tensor): 对比学习的第一个输入 ID。
        attention_mask_contrast1 (torch.Tensor): 对比学习的第一个注意力掩码。
        input_ids_contrast2 (torch.Tensor): 对比学习的第二个输入 ID。
        attention_mask_contrast2 (torch.Tensor): 对比学习的第二个注意力掩码。
        input_ids_contrast3 (torch.Tensor): 对比学习的第三个输入 ID。
        attention_mask_contrast3 (torch.Tensor): 对比学习的第三个注意力掩码。

        返回:
        torch.Tensor: 总损失。
        """
        try:
            # 计算 MLM 损失
            mlm_outputs: Any = self.base(
                input_ids=input_ids_mlm,
                attention_mask=attention_mask_mlm,
                labels=labels_mlm,
            )
            mlm_loss: torch.Tensor = mlm_outputs.loss

            # 提取对比学习所需的特征表示 (batch_size, sequence_length, hidden_size)
            contrast_output1: torch.Tensor = self.base.bert(
                input_ids=input_ids_contrast1, attention_mask=attention_mask_contrast1
            )
            contrast_output2: torch.Tensor = self.base.bert(
                input_ids=input_ids_contrast2, attention_mask=attention_mask_contrast2
            )
            contrast_output3: torch.Tensor = self.base.bert(
                input_ids=input_ids_contrast3, attention_mask=attention_mask_contrast3
            )

            # 使用平均嵌入 (batch_size, hidden_size)
            embeddings1: torch.Tensor = self._get_mean_embeddings(
                contrast_output1.last_hidden_state, attention_mask_contrast1
            )
            embeddings2: torch.Tensor = self._get_mean_embeddings(
                contrast_output2.last_hidden_state, attention_mask_contrast2
            )
            embeddings3: torch.Tensor = self._get_mean_embeddings(
                contrast_output3.last_hidden_state, attention_mask_contrast3
            )

            # 让平均嵌入经过 self.cl_head (batch_size, 256)
            embeddings1: torch.Tensor = self.cl_head(embeddings1)
            embeddings2: torch.Tensor = self.cl_head(embeddings2)
            embeddings3: torch.Tensor = self.cl_head(embeddings3)

            # 球面归一化 (batch_size, 256)
            embeddings1: torch.Tensor = F.normalize(embeddings1, p=2, dim=1)
            embeddings2: torch.Tensor = F.normalize(embeddings2, p=2, dim=1)
            embeddings3: torch.Tensor = F.normalize(embeddings3, p=2, dim=1)

            # 计算对比损失，embeddings1 是锚点，embeddings2 是正样本，embeddings3 是负样本
            contrastive_loss: torch.Tensor = self.cl_head_loss(
                embeddings1, embeddings2, embeddings3
            )

            # 总损失
            total_loss: torch.Tensor = mlm_loss + contrastive_loss
            return total_loss
        except Exception as e:
            print(f"前向传播时出错: {e}")
            raise

    # 以下是类的私有方法
    def _get_mean_embeddings(
        self, bert_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算输入序列的平均嵌入。

        参 数:
                bert_output (torch.Tensor):
                    BERT 模型最后一层的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)。
                attention_mask (torch.Tensor):
                    注意力掩码，形状为 (batch_size, sequence_length)。

        返回:
             torch.Tensor: 平均嵌入，形状为 (batch_size, hidden_size)。
        """
        # 计算每个位置的嵌入向量与注意力掩码的加权和
        weighted_sum = torch.sum(bert_output * attention_mask.unsqueeze(-1), dim=1)
        # 计算有效词元的数量
        valid_token_count = torch.sum(attention_mask.unsqueeze(-1), dim=1)
        # 计算平均嵌入
        mean_output = weighted_sum / valid_token_count
        return mean_output
