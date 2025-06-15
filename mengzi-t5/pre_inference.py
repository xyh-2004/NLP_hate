import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# 设置设备：使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 加载模型和分词器，并移动到设备上
tokenizer = T5Tokenizer.from_pretrained('best_model_1/fold0')
model = T5ForConditionalGeneration.from_pretrained('best_model_1/fold0')
model = model.to(device)
model.eval()

def predict(text):
    input_text = f"仇恨言论识别：{text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 推理并写入结果
def run_inference(input_json_path, output_txt_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc="Running inference"):
        content = item['content']
        prediction = predict(content)
        results.append(prediction.strip())

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

# 运行
if __name__ == '__main__':
    run_inference('data/test1.json', '2.txt')
