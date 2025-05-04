import random
import torch
from torch.utils.data import Dataset


class MetaFewShotDataset(Dataset):
    """
    输入：
      - data: list of (text, label) 对
      - tokenizer: HF tokenizer
      - N: ways
      - K: shots per way for support
      - Q: query shots per way
      - max_length: 文本截断长度
    __getitem__ 则随机采样一个 episode,返回：
      support_input_ids, support_attention_mask,
      query_input_ids,  query_attention_mask,
      query_labels (0..N-1)
    """

    def __init__(self, data, tokenizer, N, K, Q, max_length=64):
        super().__init__()
        # 按 label 分桶
        self.buckets = {}
        for example in data:
            text = example['text']
            label = str(example['label'])
            self.buckets.setdefault(label, []).append(text)
        self.labels = list(self.buckets.keys())
        self.tokenizer = tokenizer
        self.N, self.K, self.Q = N, K, Q
        self.max_length = max_length

    def __len__(self):
        # 任意定义个 epoch 长度
        return len(self.labels)

    def __getitem__(self, idx):
        # 1) 随机选 N 个类别
        chosen = random.sample(self.labels, self.N)
        support_texts, support_labels = [], []
        query_texts, query_labels = [], []

        while True:
            try:
                for i, label in enumerate(chosen):
                    texts = self.buckets[label]
                    sampled = random.sample(texts, self.K + self.Q)
                    support_texts += sampled[: self.K]
                    query_texts += sampled[self.K :]
                    query_labels += [i] * self.Q
                break
            except ValueError:
                chosen = random.sample(self.labels, self.N)

        # 2) 添加文本有效性校验
        support_texts = [t for t in support_texts if t.strip()]
        query_texts = [t for t in query_texts if t.strip()]

        if not support_texts or not query_texts:
            return self.__getitem__(random.randint(0, len(self)-1))

        # 3) tokenizer 编码
        support = self.tokenizer(
            support_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        query = self.tokenizer(
            query_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # 修改返回的键名以匹配模型期望
        result = {
            "support_input": support.input_ids,
            "support_mask": support.attention_mask,
            "query_input": query.input_ids,
            "query_mask": query.attention_mask,
            "labels":torch.tensor(query_labels, dtype=torch.long),
            "N": self.N,
            "K": self.K,
            "Q": self.Q,
        }
        return result

    # def collate_fn(self, batch):
    #     # 假设 batch 中只有一个元素，因为 per_device_train_batch_size 为 1
    #     if len(batch) == 0:
    #         print("警告: batch 为空")
    #         return {}
    #     print("collate_fn 输入的数据键:", batch[0].keys())
    #     if 'labels' not in batch[0]:
    #         print("警告: 输入数据中缺少 'labels' 键")
    #     result = batch[0]
    #     print("collate_fn 返回的数据键:", result.keys())
    #     return result
