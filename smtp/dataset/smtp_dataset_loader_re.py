import re
import random
import logging
import nltk
import torch
from torch import nn
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from typing import List, Dict, Tuple
from nltk.corpus import stopwords

data_dir = "data"
nltk.data.path.append(data_dir)
logging.basicConfig(level=logging.INFO)


# 数据处理策略抽象基类
class DataProcessingStrategy(ABC):
    @abstractmethod
    def process(self, dataset: Dataset) -> Dataset:
        pass


# 分词策略
class TokenizeStrategy(DataProcessingStrategy):
    def __init__(self, tokenizer_name, max_length=60) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def process(self, dataset: Dataset) -> Dataset:
        def tokenize_function(
            examples: Dict[str, str] | Dict[str, List[str]],
        ) -> Dict[str, torch.Tensor]:
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

        return dataset.map(tokenize_function, batched=True)


# 停用词替换策略
class StopwordReplaceStrategy(DataProcessingStrategy):
    def __init__(self) -> None:
        self.stopwords = self.load_stopwords()

    def load_stopwords(self) -> List[str]:
        try:
            with open("data/corpora/stopwords/english", "r", encoding="utf-8") as f:
                return [line.rstrip("\n").lstrip("\ufeff") for line in f]
        except FileNotFoundError:
            logging.error("Stopwords file not found. Please check the path.")
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", download_dir=data_dir)
            return stopwords.words("english")

    def stop_word_replace(self, text: str) -> str:
        words = re.findall(r"\w+|[^\w\s]", text)
        new_words = []
        for word in words:
            if word in self.stopwords:
                new_word = random.choice(self.stopwords)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return " ".join(new_words)

    def process(self, dataset: Dataset) -> Dataset:
        def replace_stopwords_function(
            examples: Dict[str, str] | Dict[str, List[str]],
        ) -> Dict[str, str] | Dict[str, List[str]]:
            if isinstance(examples["text"], str):
                examples["text"] = self.stop_word_replace(examples["text"])
            else:
                new_texts = []
                for text in examples["text"]:
                    new_text = self.stop_word_replace(text)
                    new_texts.append(new_text)
                examples["text"] = new_texts
            return examples

        return dataset.map(replace_stopwords_function, batched=True)


# 掩码策略
class MaskStrategy(DataProcessingStrategy):
    def __init__(self, tokenizer_name, max_length=60) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def process(self, dataset: Dataset) -> Dataset:
        def mask_function(
            examples: Dict[str, str] | Dict[str, List[str]],
        ) -> Dict[str, List[int]]:
            inputs = self.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            labels: List[int] = inputs["input_ids"].copy()
            probability_matrix: torch.Tensor = torch.full(
                torch.tensor(labels).shape, 0.15
            )
            special_tokens_mask: List[int] = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels
            ]
            probability_matrix.masked_fill_(
                torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
            )
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels = torch.tensor(labels)
            labels[~masked_indices] = -100

            indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            )
            input_ids = torch.tensor(inputs["input_ids"])
            input_ids[indices_replaced] = self.tokenizer.mask_token_id

            indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
            )
            random_words = torch.randint(
                len(self.tokenizer), labels.shape, dtype=torch.long
            )
            input_ids[indices_random] = random_words[indices_random]

            inputs["input_ids"] = input_ids.tolist()
            inputs["labels"] = labels.tolist()
            return inputs

        return dataset.map(mask_function, batched=True)


# 数据打乱策略
class ShuffleStrategy(DataProcessingStrategy):
    def process(self, dataset: Dataset) -> Dataset:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        return dataset.select(indices)


# 数据处理器
class DataProcessor:
    def __init__(self) -> None:
        self.strategies = []

    def add_strategy(self, strategy: DataProcessingStrategy):
        self.strategies.append(strategy)

    def process(self, dataset: Dataset) -> Dataset:
        for strategy in self.strategies:
            dataset = strategy.process(dataset)
        return dataset


def merge_datasets(
    original: Dataset, stopwords: Dataset, shuffled: Dataset, masked: Dataset
) -> Dataset:
    new_dataset = {
        "input_ids_mlm": masked["input_ids"],
        "attention_mask_mlm": masked["attention_mask"],
        "labels_mlm": masked["labels"],
        "input_ids_contrast1": original["input_ids"],
        "attention_mask_contrast1": original["attention_mask"],
        "input_ids_contrast2": stopwords["input_ids"],
        "attention_mask_contrast2": stopwords["attention_mask"],
        "input_ids_contrast3": shuffled["input_ids"],
        "attention_mask_contrast3": shuffled["attention_mask"],
    }
    return Dataset.from_dict(new_dataset)


# 加载数据集
def loader_dataset(datasetname, tokenizer_name) -> Dataset:
    datasets = []
    for name in datasetname:
        path = f"data/{name}/train.csv"
        try:
            dataset = load_dataset("csv", data_files=path)["train"]
            print(f"Columns in {path}: {dataset.column_names}")  # 打印列名

            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col != "text"]
            )
            datasets.append(dataset)
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")

    if not datasets:
        print("No valid datasets were loaded.")
        return None

    combined_dataset = (
        datasets[0]
        if len(datasets) == 1
        else datasets[0].concatenate_datasets(datasets[1:])
    )

    # 初始化数据处理器和策略
    tokenize_strategy = TokenizeStrategy(tokenizer_name)
    stopword_replace_strategy = StopwordReplaceStrategy()
    mask_strategy = MaskStrategy(tokenizer_name)
    shuffle_strategy = ShuffleStrategy()

    # 处理原始数据
    processor_original = DataProcessor()
    processor_original.add_strategy(tokenize_strategy)
    original_dataset = processor_original.process(combined_dataset)

    # 处理停用词替换数据
    processor_stopwords = DataProcessor()
    processor_stopwords.add_strategy(stopword_replace_strategy)
    processor_stopwords.add_strategy(tokenize_strategy)
    stopwords_dataset = processor_stopwords.process(combined_dataset)

    # 处理随机打乱数据
    processor_shuffled = DataProcessor()
    processor_shuffled.add_strategy(shuffle_strategy)
    processor_shuffled.add_strategy(tokenize_strategy)
    shuffled_dataset = processor_shuffled.process(combined_dataset)

    # 处理掩码数据
    processor_masked = DataProcessor()
    processor_masked.add_strategy(mask_strategy)
    masked_dataset = processor_masked.process(combined_dataset)

    merged_dataset = merge_datasets(
        original_dataset, stopwords_dataset, shuffled_dataset, masked_dataset
    )
    return merged_dataset
