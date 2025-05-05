from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
import torch.nn.functional as F
import evaluate


class MetaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, compute_metrics=self.compute_metrics, **kwargs)
        # 初始化评估指标
        self.accuracy_metric = evaluate.load("accuracy")  # 准确率
        self.precision_metric = evaluate.load("precision")  # 精确率
        self.recall_metric = evaluate.load("recall")  # 召回率
        self.f1_metric = evaluate.load("f1")  # F1 分数

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) 取出标签
        labels = inputs["labels"].squeeze(0)
        # 2) forward
        out = model(
            inputs["support_input"],
            inputs["support_mask"],
            inputs["query_input"],
            inputs["query_mask"],
        )

        logits = out["logits"]
        # 3) CE
        loss_ce = F.cross_entropy(logits, labels)
        # 4) UCL1
        p1, p2 = F.normalize(out["proto1"], dim=1), F.normalize(out["proto2"], dim=1)
        loss_ucl1 = (1 - (p1 * p2).sum(dim=1)).mean()
        # 5) UCL2
        q1, q2 = F.normalize(out["xq1"], dim=1), F.normalize(out["xq2"], dim=1)
        loss_ucl2 = (1 - (q1 * q2).sum(dim=1)).mean()
        loss = loss_ce + loss_ucl1 + loss_ucl2
        return (loss, out) if return_outputs else loss

    def compute_metrics(self, eval_pred: EvalPrediction):
        # 处理元组类型的 predictions
        if isinstance(eval_pred.predictions, tuple):
            print(f"检测到元组类型的 predictions,包含 {len(eval_pred.predictions)} 个元素")
            logits = eval_pred.predictions[0]  # 提取第一个元素作为 logits
        else:
            logits = eval_pred.predictions
        
        # 处理二维标签数组（原始代码中的问题）
        labels = eval_pred.label_ids.reshape(-1)  # 展平为 1D 数组

        preds = logits.argmax(axis=-1)

        # 2) 计算各项指标
        acc = self.accuracy_metric.compute(predictions=preds, references=labels)[
            "accuracy"
        ]
        prec = self.precision_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["precision"]
        rec = self.recall_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["recall"]
        f1 = self.f1_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["f1"]
        return {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        }
