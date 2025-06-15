import torch
from torch.utils.data import Dataset
import numpy as np
from config import TARGETED_GROUPS, MAX_LEN

def group_to_multi_hot(groups):
    multi_hot = np.zeros(len(TARGETED_GROUPS), dtype=np.float32)
    for g in groups:
        if g in TARGETED_GROUPS:
            multi_hot[TARGETED_GROUPS.index(g)] = 1.0
    return multi_hot

class HateSpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        import json
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["content"]  # 读取文本

        # 解析 output 字符串成四元组标签列表
        output_str = sample.get("output", "")
        labels = []
        if output_str:
            tuples = output_str.strip().split("[SEP]")
            for tup_str in tuples:
                tup_str = tup_str.strip()
                if not tup_str:
                    continue
                parts = tup_str.split("|")
                if len(parts) == 4:
                    target = parts[0].strip()
                    argument = parts[1].strip()
                    group_str = parts[2].strip()
                    hateful_str = parts[3].strip()

                    # group 和 hateful 处理
                    if group_str.lower() == "non-hate":
                        group = []
                    else:
                        group = [group_str]

                    hateful = 0 if hateful_str.lower() == "non-hate" else 1

                    labels.append({
                        "target": target,
                        "argument": argument,
                        "group": group,
                        "hateful": hateful
                    })

        encoding = self.tokenizer(text,
                                  max_length=MAX_LEN,
                                  padding="max_length",
                                  truncation=True,
                                  return_offsets_mapping=True,
                                  return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)

        target_start_labels = torch.zeros(MAX_LEN, dtype=torch.float)
        target_end_labels = torch.zeros(MAX_LEN, dtype=torch.float)
        arg_start_labels = torch.zeros(MAX_LEN, dtype=torch.float)
        arg_end_labels = torch.zeros(MAX_LEN, dtype=torch.float)

        group_labels = torch.zeros(len(TARGETED_GROUPS), dtype=torch.float)
        hateful_labels = torch.zeros(1, dtype=torch.float)

        # 根据标签在文本中查找对应的token范围，标记start/end
        for quad in labels:
            target = quad.get("target", "")
            argument = quad.get("argument", "")
            groups = quad.get("group", [])
            hateful = quad.get("hateful", 0)

            # target对应的span标注
            target_start, target_end = self._char_span_to_token_span(text, target, offsets)
            if target_start is not None:
                target_start_labels[target_start] = 1.0
                target_end_labels[target_end] = 1.0

            # argument对应的span标注
            arg_start, arg_end = self._char_span_to_token_span(text, argument, offsets)
            if arg_start is not None:
                arg_start_labels[arg_start] = 1.0
                arg_end_labels[arg_end] = 1.0

            group_hot = group_to_multi_hot(groups)
            group_labels = torch.maximum(group_labels, torch.tensor(group_hot))

            hateful_labels[0] = max(hateful_labels[0], hateful)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_start_labels": target_start_labels,
            "target_end_labels": target_end_labels,
            "arg_start_labels": arg_start_labels,
            "arg_end_labels": arg_end_labels,
            "group_labels": group_labels,
            "hateful_labels": hateful_labels
        }

    def _char_span_to_token_span(self, text, span_text, offsets):
        if not span_text or span_text == "NULL":
            return None, None

        start_char_idx = text.find(span_text)
        if start_char_idx == -1:
            return None, None
        end_char_idx = start_char_idx + len(span_text)

        token_start_idx, token_end_idx = None, None
        for i, (start_offset, end_offset) in enumerate(offsets.tolist()):
            if start_offset <= start_char_idx < end_offset:
                token_start_idx = i
            if start_offset < end_char_idx <= end_offset:
                token_end_idx = i

        # 如果end找不到，取最后一个包含end_char_idx的token
        if token_start_idx is not None and token_end_idx is None:
            for i, (start_offset, end_offset) in enumerate(offsets.tolist()):
                if end_offset >= end_char_idx:
                    token_end_idx = i
                    break

        if token_start_idx is not None and token_end_idx is not None:
            return token_start_idx, token_end_idx
        else:
            return None, None
