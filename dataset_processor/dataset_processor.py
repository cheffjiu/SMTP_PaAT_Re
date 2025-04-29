import os
import pandas as pd
import logging
import subprocess
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from typing import Dict, Optional, Union
import random

logging.basicConfig(level=logging.INFO)


def download_liu57() -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/jianguoz/Few-Shot-Intent-Detection.git",
            "data/liu57",
        ],
        check=True,
    )  # 克隆到 data/liu57


class DatasetProcessor:
    def __init__(self, field_mapping: Optional[Dict] = None) -> None:
        # 确保初始化field_mapping属性
        self.field_mapping = field_mapping or {
            "utterance": "text",  # 修复映射关系
            "text": "text",  # 新增保留原始text字段
            "sentence": "text",
            "intent": "label",
            "target": "label",
        }

    def load_dataset(self, format: str, path: str) -> DatasetDict | Dataset:
        """加载本地数据集，支持多种本地格式"""
        try:
            logging.info(f"开始加载数据集: {path}")
            # 优先尝试加载Arrow格式
            if format == "arrow":
                dataset: Union[DatasetDict, Dataset] = load_from_disk(path)
                logging.info(f"数据集加载完成(Arrow):{path}")
                print(dataset)
                return dataset

            if format == "csv":
                data_files: Dict[str, str] = {
                    split: f"{path}/{split}.csv"
                    for split in ["train", "test", "validation"]
                    if os.path.exists(f"{path}/{split}.csv")
                }
                dataset: Union[DatasetDict, Dataset] = load_dataset(
                    "csv", data_files=path
                )
                logging.info(f"数据集加载完成(CSV):{path}")
                return dataset
        except Exception as e:
            logging.error(f"加载数据集失败: {e}")
            raise

    def convert_format(
        self, dataname: str, input_path: str, output_path: str, conversion_type: str
    ) -> None:
        """增强版格式转换
        支持转换类型：
        - arrow_to_csv : Arrow格式 → CSV
        - csv_to_arrow : CSV → Arrow格式
        - seq_to_csv   : (seq.in + label) → CSV
        """
        try:
            if conversion_type == "arrow_to_csv":
                self._arrow_to_csv(dataname, input_path, output_path)
            elif conversion_type == "csv_to_arrow":
                self._csv_to_arrow(input_path, output_path)
            elif conversion_type == "seq_to_csv":
                self._seq_to_csv(input_path, output_path)
            else:
                raise ValueError(f"不支持的转换类型: {conversion_type}")
        except Exception as e:
            logging.error(f"格式转换失败: {e}")
            raise

    def merge_splits(self, input_path: str, output_path: str) -> None:
        """
        将数据集的各个划分（如训练集、测试集、验证集）合并为一个整体数据集。
        :param input_path: 输入数据集的路径
        :param output_path: 合并后数据集的保存路径
        """
        dataset = self.load_dataset("arrow", input_path)
        if isinstance(dataset, DatasetDict):
            combined_data = {}
            for split in dataset:
                for col in dataset[split].column_names:
                    if col not in combined_data:
                        combined_data[col] = []
                    combined_data[col].extend(dataset[split][col])
            combined_dataset = Dataset.from_dict(combined_data)
        else:
            combined_dataset = dataset

        combined_dataset = self._rename_features(combined_dataset)
        self._save_local(combined_dataset, output_path)
        logging.info(f"数据集已合并并保存至 {output_path}")

    def split_meta_dataset(
        self,
        dataset: Dataset | DatasetDict,
        test_size: float = 0.2,
        save_path: Optional[str] = None,
        random_seed: int = 42,
    ) -> DatasetDict:
        """
        将数据集按label划分为元训练集和元测试集,保证两者label不相交。
        :param dataset: 输入的数据集，可以是 Dataset 或 DatasetDict 类型
        :param test_size: 元测试集的label比例,默认 0.2
        :param save_path: 保存划分后数据集的路径，可选
        :param random_seed: 随机种子，用于确保划分结果可复现，默认 42
        :return: 包含元训练集和元测试集的 DatasetDict
        """
        if isinstance(dataset, DatasetDict):
            # 如果是 DatasetDict，先合并所有划分
            combined_data = {}
            for split in dataset:
                for col in dataset[split].column_names:
                    if col not in combined_data:
                        combined_data[col] = []
                    combined_data[col].extend(dataset[split][col])
            dataset = Dataset.from_dict(combined_data)

        # 设置随机种子
        random.seed(random_seed)

        # 获取所有唯一的label
        unique_labels = list(set(dataset["label"]))
        # 打乱label顺序
        random.shuffle(unique_labels)
        # 计算元测试集的label数量
        test_label_count = int(len(unique_labels) * test_size)
        # 划分label
        test_labels = unique_labels[:test_label_count]
        train_labels = unique_labels[test_label_count:]

        # 根据label划分数据集
        meta_train = dataset.filter(lambda example: example["label"] in train_labels)
        meta_test = dataset.filter(lambda example: example["label"] in test_labels)

        meta_dataset = DatasetDict({"meta_train": meta_train, "meta_test": meta_test})

        if save_path:
            self._save_local(meta_dataset, save_path)
            logging.info(f"元训练集和元测试集已保存至 {save_path}")

        return meta_dataset

    # 以下是内部方法
    def _save_local(self, dataset: Dataset | DatasetDict, path: str) -> None:
        """保存数据集到本地"""
        try:
            os.makedirs(path, exist_ok=True)
            dataset.save_to_disk(path)
            logging.info(f"数据集已保存至: {path}")
        except Exception as e:
            logging.error(f"保存数据集失败: {e}")
            raise

    def _rename_features(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """修复后的字段重命名方法"""
        if isinstance(dataset, DatasetDict):
            return DatasetDict(
                {
                    split: self._rename_features(subset)
                    for split, subset in dataset.items()
                }
            )

        # 动态生成有效映射
        effective_mapping = {
            src: dest
            for src, dest in self.field_mapping.items()
            if src in dataset.column_names
        }
        return dataset.rename_columns(effective_mapping)

    def _arrow_to_csv(
        self, datasetname: str, input_path: str, output_path: str
    ) -> None:
        dataset = self.load_dataset("arrow", input_path)

        # 添加类型注解确保正确访问属性
        if isinstance(dataset, DatasetDict):
            dataset = DatasetDict(
                {
                    split: self._rename_features(subset)
                    for split, subset in dataset.items()
                }
            )
        else:
            dataset = self._rename_features(dataset)

        os.makedirs(output_path, exist_ok=True)
        if isinstance(dataset, DatasetDict):
            for split in dataset:  # split: train/test/validation
                dataset[split].to_csv(f"{output_path}/{split}.csv")
        elif isinstance(dataset, Dataset):
            dataset.to_csv(f"{output_path}/{datasetname}.csv")
        logging.info("数据集已转换为CSV格式")

    def _csv_to_arrow(self, input_path: str, output_path: str) -> None:
        """将CSV格式数据集转换为Arrow格式"""
        dataset: Union[DatasetDict, Dataset] = self.load_dataset("csv", input_path)
        os.makedirs(output_path, exist_ok=True)
        self._save_local(dataset, output_path)

    def _seq_to_csv(self, split: str, input_path: str, output_path: str) -> None:
        """序列文件转 CSV 实现"""
        texts = open(f"{input_path}/seq.in").read().splitlines()
        labels = open(f"{input_path}/label").read().splitlines()
        pd.DataFrame({"text": texts, "label": labels}).to_csv(
            f"{output_path}/{split}.csv", index=False
        )


if __name__ == "__main__":
    processor = DatasetProcessor()

    # 数据集下载
    # download_liu57()
    banking77 = load_dataset("PolyAI/banking77")
    banking77.save_to_disk("data/banking77")
    hwu64 = load_dataset("DeepPavlov/hwu64")
    hwu64.save_to_disk("data/hwu64")
    clincl150 = load_dataset("DeepPavlov/clinc150")
    clincl150.save_to_disk("data/clinc150")

    # 数据集合并
    processor.merge_splits("data/hwu64", "data/hwu64_merge")
    processor.merge_splits("data/clinc150", "data/clinc150_merge")
    processor.merge_splits("data/banking77", "data/banking77_merge")

    # 格式转换
    processor.convert_format(
        "hwu64", "data/hwu64_merge/", "data/hwu64_csv", "arrow_to_csv"
    )
    processor.convert_format(
        "banking77", "data/banking77_merge", "data/banking77_csv", "arrow_to_csv"
    )
    processor.convert_format(
        "clinc150", "data/clinc150_merge", "data/clinc150_csv", "arrow_to_csv"
    )
    # 元训练集和元测试集划分
    banking77_meta = processor.load_dataset("csv", "data/banking77_csv/banking77.csv")
    processor.split_meta_dataset(
        banking77_meta, save_path="data/banking77_meta", random_seed=42
    )
    processor.convert_format(
        "banking77", "data/banking77_meta", "data/banking77_meta_csv", "arrow_to_csv"
    )

    clinc150_meta = processor.load_dataset("csv", "data/clinc150_csv/clinc150.csv")
    processor.split_meta_dataset(
        clinc150_meta, save_path="data/clinc150_meta", random_seed=42
    )
    processor.convert_format(
        "clinc150", "data/clinc150_meta", "data/clinc150_meta_csv", "arrow_to_csv"
    )

    hwu64_meta = processor.load_dataset("csv", "data/hwu64_csv/hwu64.csv")
    processor.split_meta_dataset(
        hwu64_meta, save_path="data/hwu64_meta", random_seed=42
    )
    processor.convert_format(
        "hwu64", "data/clinc150_meta", "data/clinc150_meta_csv", "arrow_to_csv"
    )
