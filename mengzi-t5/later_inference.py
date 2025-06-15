import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 加载训练时保存的最优模型（来自最终复制到 best_model_overall 的路径）
model_path = 'best_model_1/fold0'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)
model.eval()

# 预测函数
def predict(text):
    # 与训练输入保持一致
    input_text = f"任务：抽取仇恨言论四元组（评论对象 | 论点 | 目标群体 | hate/non-hate），文本如下：{text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# 推理主函数
def run_inference(input_json_path, output_txt_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc="正在进行推理"):
        content = item['content']
        pred = predict(content)
        results.append(pred)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

# 运行
if __name__ == '__main__':
    run_inference('./data/test1.json', './predictions.txt')
