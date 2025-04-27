from transformers import TrainingArguments
from model.smtp_bert_re import SMTPBert
from dataset.smtp_dataset_loader_re import loader_dataset
from trainer.smtp_trainer_re import CustomTrainer


# 主函数
if __name__ == "__main__":

    class Args:
        def __init__(self):
            self.dataname = ["banking77_csv"]
            self.output_dir = "module"
            self.num_train_epochs = 3
            self.per_device_train_batch_size = 16

    args = Args()
    tokenizer_name = "bert-base-uncased"
    model_name = "bert-base-uncased"

    # 加载数据集
    merged_dataset = loader_dataset(args, tokenizer_name)

    # 初始化模型
    model = SMTPBert(model_name)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    # 初始化 Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=merged_dataset,
    )

    # 开始训练
    trainer.train()
