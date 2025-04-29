
---

# Few-Shot Intent Detection Dataset Processor

用于下载、合并、转换和划分用于 Few-Shot 意图识别任务的常用数据集（如 `banking77`、`clinc150`、`hwu64`、`liu57`），并统一成标准格式，便于后续训练和评估。

## 📦 功能概览

- 自动下载常用意图识别数据集
- 支持多格式数据加载（Arrow、CSV、自定义序列格式）
- 支持数据格式转换（如 Arrow → CSV）
- 支持将多划分数据集合并为整体
- 支持按标签无交集方式划分元训练集/元测试集
- 支持字段统一重命名（如 `utterance` / `sentence` → `text`，`intent` / `target` → `label`）

---

## 📁 项目结构

```bash
.
├── dataset_processor.py         # 主处理脚本
├── data/
│   ├── liu57/                   # 下载的 Few-Shot-Intent-Detection repo
│   ├── banking77/               # 原始 banking77 数据集（Arrow 格式）
│   ├── banking77_merge/         # 合并后的 banking77 数据
│   ├── banking77_csv/           # 转换后的 CSV 格式
│   ├── banking77_meta/          # 元训练/测试集（Arrow 格式）
│   └── banking77_meta_csv/      # 元训练/测试集（CSV 格式）
└── ...
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install datasets pandas
```

### 2. 运行脚本

默认流程包含以下步骤：

- 下载数据集：`banking77`、`clinc150`、`hwu64`
- 将三类数据集合并为整体
- 转换为 CSV 格式
- 划分为元训练集 / 元测试集（保证 label 不重合）
- 保存划分结果并转换为 CSV 格式

```bash
python dataset_processor.py
```

> 如果需要克隆 `liu57` 数据集，请取消 `download_liu57()` 的注释。

---

## 🧠 主要类与方法

### `DatasetProcessor`

| 方法名                                  | 功能描述                                                            |
| --------------------------------------- | ------------------------------------------------------------------- |
| `load_dataset(format, path)`            | 加载 Arrow/CSV 格式数据                                             |
| `convert_format(...)`                   | 数据格式转换（如 Arrow ↔ CSV，seq ↦ CSV）                           |
| `merge_splits(input_path, output_path)` | 合并 train/test/validation 数据为一个整体                           |
| `split_meta_dataset(...)`               | 将数据划分为 meta-train / meta-test，label 不交叉                   |
| `_rename_features(...)`                 | 将 `utterance` / `sentence` → `text`，`intent` / `target` → `label` |

---

## 📝 说明

- **字段重命名规则**：
  - `"utterance"`, `"sentence"`, `"text"` → `"text"`
  - `"intent"`, `"target"` → `"label"`
- **划分策略**：元测试集中包含 20% 的 label，剩余用于元训练，确保二者 label 无交集。

---

## 🧩 数据来源

- [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- [DeepPavlov/clinc150](https://huggingface.co/datasets/DeepPavlov/clinc150)
- [DeepPavlov/hwu64](https://huggingface.co/datasets/DeepPavlov/hwu64)
- [Few-Shot-Intent-Detection (Liu et al.)](https://github.com/jianguoz/Few-Shot-Intent-Detection)

---

## 📌 TODO

- [ ] 增加对 Huggingface DatasetDict 的上传与远程保存支持
- [ ] 支持自定义分词与标准化
- [ ] 添加元训练集/测试集可视化模块

---

