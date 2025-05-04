from transformers import Trainer
import torch.nn.functional as F


class MetaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) 取出标签 & 超参
        labels = inputs["labels"].squeeze(0)
        N, K, Q = inputs.pop("N"), inputs.pop("K"), inputs.pop("Q")
        # 2) forward
        out = model(inputs["support_input"],inputs["support_mask"], inputs["query_input"], inputs["query_mask"],labels,N, K, Q)
        logits = out["logits"]
        # 3) CE
        loss_ce = F.cross_entropy(
            logits, labels
        )  # contentReference[oaicite:5]{index=5}
        # 4) UCL1
        p1, p2 = F.normalize(out["proto1"], dim=1), F.normalize(out["proto2"], dim=1)
        loss_ucl1 = (
            1 - (p1 * p2).sum(dim=1)
        ).mean()  # contentReference[oaicite:6]{index=6}
        # 5) UCL2
        q1, q2 = F.normalize(out["xq1"], dim=1), F.normalize(out["xq2"], dim=1)
        loss_ucl2 = (
            1 - (q1 * q2).sum(dim=1)
        ).mean()  # contentReference[oaicite:7]{index=7}
        loss = loss_ce + loss_ucl1 + loss_ucl2
        return (
            (loss, out) if return_outputs else loss
        )  # contentReference[oaicite:8]{index=8}
