import torch
from transformers import BertTokenizerFast
from hate_span_extractor import HateSpeechTupleExtractor
from utils import HateSpeechDataset
from config import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def compute_iou(str_a, str_b):
    set_a = set(str_a.strip().split())
    set_b = set(str_b.strip().split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0

def inference():
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    model = HateSpeechTupleExtractor(PRETRAINED_MODEL).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()

    with open("data/test1.json", "r", encoding="utf-8") as f:
        infer_data = json.load(f)

    dataset = HateSpeechDataset("data/test1.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    all_output_lines = []
    group_labels = ["Region", "Racism", "Sexism", "LGBTQ", "others", "non-hate"]
    hate_labels = ["non-hate", "hate"]

    seen_results = set()              # 四元组级去重
    seen_arg_map = dict()            # {arg_text: tgt_text} 防止相同论点跨主体复用
    seen_args_per_target = dict()    # {tgt_text: [arg_text1, ...]} 防止主体内部重复论点

    with torch.no_grad():
        for batch, data_item in tqdm(zip(dataloader, infer_data), total=len(infer_data), desc="Running inference"):
            batch = batch[0] if isinstance(batch, list) else batch
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            start_tgt, end_tgt, start_arg, end_arg, group_logits, hateful_logits = model(input_ids, attention_mask)
            start_tgt = sigmoid(start_tgt).squeeze(0)
            end_tgt = sigmoid(end_tgt).squeeze(0)
            start_arg = sigmoid(start_arg).squeeze(0)
            end_arg = sigmoid(end_arg).squeeze(0)

            tgt_starts = (start_tgt > 0.5).nonzero(as_tuple=True)[0]
            tgt_ends = (end_tgt > 0.5).nonzero(as_tuple=True)[0]
            arg_starts = (start_arg > 0.5).nonzero(as_tuple=True)[0]
            arg_ends = (end_arg > 0.5).nonzero(as_tuple=True)[0]

            example_outputs = []

            for t_s in tgt_starts:
                t_e_candidates = tgt_ends[tgt_ends >= t_s]
                if len(t_e_candidates) == 0:
                    continue
                t_e = t_e_candidates[0]
                tgt_tokens = input_ids[0][t_s:t_e + 1]
                tgt_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True).replace(" ", "").strip()
                if not tgt_text:
                    continue

                for a_s in arg_starts:
                    a_e_candidates = arg_ends[arg_ends >= a_s]
                    if len(a_e_candidates) == 0:
                        continue
                    a_e = a_e_candidates[0]
                    arg_tokens = input_ids[0][a_s:a_e + 1]
                    arg_text = tokenizer.decode(arg_tokens, skip_special_tokens=True).replace(" ", "").strip()
                    if not arg_text:
                        continue

                    # ==== 去重逻辑 ====
                    is_duplicate = False
                    hash_key = f"{tgt_text}|{arg_text}"

                    # 1. 完全重复
                    if hash_key in seen_results:
                        continue

                    # 2. 相同论点不能复用到不同主体
                    for seen_arg, seen_tgt in seen_arg_map.items():
                        if compute_iou(arg_text, seen_arg) > 0.8 and tgt_text != seen_tgt:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        continue

                    # 3. 同一主体已有相似论点
                    if tgt_text in seen_args_per_target:
                        for seen_arg in seen_args_per_target[tgt_text]:
                            if compute_iou(arg_text, seen_arg) > 0.5:
                                is_duplicate = True
                                break
                    if is_duplicate:
                        continue

                    # ==== 标签预测 ====
                    try:
                        group_label = group_labels[group_logits.argmax().item()]
                        hate_label = hate_labels[hateful_logits.argmax().item()]
                    except Exception as e:
                        print(f"[ERROR] label decoding failed: {e}")
                        group_label = "others"
                        hate_label = "non-hate"

                    result = f"{tgt_text} | {arg_text} | {group_label} | {hate_label} [END]"
                    example_outputs.append(result)

                    # 更新记录
                    seen_results.add(hash_key)
                    seen_arg_map[arg_text] = tgt_text
                    if tgt_text not in seen_args_per_target:
                        seen_args_per_target[tgt_text] = []
                    seen_args_per_target[tgt_text].append(arg_text)

            line_output = " [SEP] ".join(example_outputs)
            all_output_lines.append(line_output)

    with open("inference_output.txt", "w", encoding="utf-8") as f:
        for line in all_output_lines:
            f.write(line + "\n")


    print(f"[SAVED] Total lines written: {len(all_output_lines)} (+1 empty line)")

if __name__ == "__main__":
    inference()
