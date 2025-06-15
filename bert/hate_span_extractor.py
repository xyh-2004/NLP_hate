import torch
import torch.nn as nn
from transformers import BertModel
from config import TARGETED_GROUPS

class SpanExtractor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output):
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)
        return start_logits, end_logits


class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_groups):
        super().__init__()
        self.group_classifier = nn.Linear(hidden_size, num_groups)
        self.hate_classifier = nn.Sequential(
            nn.Linear(hidden_size + num_groups, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # binary: non-hate / hate
        )

    def forward(self, cls_repr):
        # 先用 CLS 表征做 group 分类
        group_logits = self.group_classifier(cls_repr)  # [B, num_groups]
        group_probs = torch.softmax(group_logits, dim=-1)

        # 拼接 group 预测分布作为输入给 hate 分类器
        hate_input = torch.cat([cls_repr, group_probs], dim=-1)
        hate_logits = self.hate_classifier(hate_input)  # [B, 2]

        return group_logits, hate_logits

class HateSpeechTupleExtractor(nn.Module):
    def __init__(self, pretrained_model_name, num_groups=len(TARGETED_GROUPS)):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        self.target_start = nn.Linear(hidden_size, 1)
        self.target_end = nn.Linear(hidden_size, 1)
        self.arg_start = nn.Linear(hidden_size, 1)
        self.arg_end = nn.Linear(hidden_size, 1)

        self.classifier_head = ClassifierHead(hidden_size, num_groups)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # span extraction logits
        tgt_start_logits = self.target_start(sequence_output).squeeze(-1)
        tgt_end_logits = self.target_end(sequence_output).squeeze(-1)
        arg_start_logits = self.arg_start(sequence_output).squeeze(-1)
        arg_end_logits = self.arg_end(sequence_output).squeeze(-1)

        # 使用 CLS 表征进行分类
        cls_repr = sequence_output[:, 0, :]  # [CLS] token
        group_logits, hate_logits = self.classifier_head(cls_repr)

        return tgt_start_logits, tgt_end_logits, arg_start_logits, arg_end_logits, group_logits, hate_logits
