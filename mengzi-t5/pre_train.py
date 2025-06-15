import json
import torch
import numpy as np
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import EarlyStoppingCallback

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs, targets = [], []
    for item in data:
        content = item['content']
        output = item['output']
        inputs.append(f"仇恨言论识别：{content}")
        targets.append(output)
    return Dataset.from_dict({'input': inputs, 'target': targets})


# 预处理函数
def preprocess_function(examples):
    model_inputs = tokenizer(examples['input'], max_length=256, truncation=True, padding='max_length')
    labels = tokenizer(examples['target'], max_length=64, truncation=True, padding='max_length')
    model_inputs['labels'] = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in label]
        for label in labels['input_ids']
    ]
    # 确保 labels 是整数张量
    return model_inputs


# 安全处理 decode 嵌套的函数
def flatten_nested(pred):
    # 将 list 嵌套结构平展为 List[List[int]]
    def _flatten(batch):
        if isinstance(batch, list) and all(isinstance(i, list) for i in batch):
            if all(isinstance(j, list) for i in batch for j in i):  # 三层嵌套
                return [j for i in batch for j in i]
            else:
                return batch
        return batch

    return _flatten(pred)


# 计算评价指标
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # 确保 preds 和 labels 是整数张量
    preds = preds.argmax(-1)  # 如果 preds 是概率分布，取 argmax 转换为 token IDs
    preds = preds.flatten()  # 转换为一维 numpy 数组
    labels = labels.flatten()  # 转换为一维 numpy 数组

    def batch_decode(ids, batch_size=16):
        decoded = []
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            decoded += tokenizer.batch_decode(batch, skip_special_tokens=True)
        return [s.strip() for s in decoded]

    decoded_preds = batch_decode(preds)
    decoded_labels = batch_decode(labels)

    acc = accuracy_score(decoded_labels, decoded_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        decoded_labels, decoded_preds, average='macro', zero_division=0
    )

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    print("当前设备：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    dataset = load_data('data/train.json')
    fold_num = 5
    fold_size = len(dataset) // fold_num

    for fold in range(fold_num):
        print(f"\n========== Fold {fold + 1} / {fold_num} ==========")
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold != fold_num - 1 else len(dataset)

        val_dataset = dataset.select(range(val_start, val_end))
        train_indices = list(set(range(len(dataset))) - set(range(val_start, val_end)))
        train_dataset = dataset.select(train_indices)

        global tokenizer
        tokenizer = T5Tokenizer.from_pretrained('mengzi-t5-base')
        model = T5ForConditionalGeneration.from_pretrained('mengzi-t5-base')

        train_dataset = train_dataset.map(preprocess_function, batched=True)
        val_dataset = val_dataset.map(preprocess_function, batched=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=f'./results_1/{fold}',
            evaluation_strategy='epoch',
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy='epoch',
            logging_dir=f'./logs_1/{fold}',
            logging_steps=10,
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            fp16=True,
            lr_scheduler_type='cosine',  # 或 'linear'
            warmup_ratio=0.1,
            num_train_epochs=10
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        trainer.save_model(f'./saved_model_1/{fold}')


if __name__ == '__main__':
    main()
