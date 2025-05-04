import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class FIATLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        pooled = attn_out.max(dim=1, keepdim=True)[0]  # [B,1,hidden_size]
        pooled = pooled.expand(-1, x.size(1), -1)  # [B,L,hidden_size]
        return self.layernorm(pooled)  # [B,L,hidden_size]


class PaLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, proto):
        score = self.tanh(self.W(proto))  # [N,L,hidden_size]
        weight = self.softmax(score)  # [N,L,hidden_size]
        return proto * weight  # [N,L,hidden_size]


class AdaptiveLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.LayerNorm(hidden_size)
        )

    def forward(self, x):
        pooled = x.max(dim=1)[0]  # [B,hidden_size]
        return self.ffn(pooled)  # [B,hidden_size]


class PaATModel(nn.Module):
    def __init__(self, pretrained_model_name: str, temp: float = 1.0):
        super().__init__()
        cfg = AutoConfig.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=cfg)
        hidden_size = cfg.hidden_size
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.fiat = FIATLayer(hidden_size, num_heads=cfg.num_attention_heads)
        self.pa = PaLayer(hidden_size)
        self.ad = AdaptiveLayer(hidden_size)
        self.temp = temp

    def make_protos(self, support_seq, N, K):
        # support_seq: [N*K, L, hidden_size]
        support_seq = support_seq.view(N, K, support_seq.size(1), support_seq.size(2))  # [N,K,L,hidden_size]
        # inner-sent
        x_sent = support_seq.view(N * K, support_seq.size(2), support_seq.size(3))  # [N*K,L,hidden_size]
        I_sent = self.fiat(x_sent).view(N, K, support_seq.size(2), support_seq.size(3))
        # inner-class
        x_cls = support_seq.permute(0, 2, 1, 3).reshape(
            N * support_seq.size(2), K, support_seq.size(3)
        )
        I_cls = self.fiat(x_cls).view(N, support_seq.size(2), support_seq.size(3))
        proto1 = 0.5 * (I_sent.mean(dim=1) + I_cls)  # [N,L,hidden_size]
        proto2 = self.pa(proto1)  # [N,L,hidden_size]
        return self.ad(proto2)  # [N,hidden_size]

    def make_queries(self, qry_seq):
        # qry_seq: [N*Q, L, hidden_size]
        q_sent = self.fiat(qry_seq)  # [N*Q,L,hidden_size]
        return self.ad(q_sent)  # [N*Q,hidden_size]

    def forward(self, support_input, support_mask, query_input, query_mask,labels ,N, K, Q):
        support_input=support_input.squeeze(0)
        support_mask=support_mask.squeeze(0)
        query_input=query_input.squeeze(0)
        query_mask=query_mask.squeeze(0)
        # 1) 编码
        sup = self.encoder(input_ids=support_input, attention_mask=support_mask)[0]
        qry = self.encoder(input_ids=query_input, attention_mask=query_mask)[0]
        # 2) dropout 版本
        sup1, sup2 = sup, self.dropout(sup)
        qry1, qry2 = qry, self.dropout(qry)
        # 3) 构造原型 & 查询向量
        proto1 = self.make_protos(sup1, N, K)
        proto2 = self.make_protos(sup2, N, K)
        xq1 = self.make_queries(qry1)
        xq2 = self.make_queries(qry2)
        # 4) logits
        p1 = F.normalize(proto1, dim=1)  # [N,hidden_size]
        q1 = F.normalize(xq1, dim=1)  # [N*Q,hidden_size]
        logits = torch.matmul(q1, p1.T) / self.temp  # [N*Q, N]
        return {
            "logits": logits,
            "proto1": proto1,
            "proto2": proto2,
            "xq1": xq1,
            "xq2": xq2,
        }
