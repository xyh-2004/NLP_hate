import json
import torch
from transformers import BertTokenizerFast, AdamW
from torch.utils.data import DataLoader, Subset
from hate_span_extractor import HateSpeechTupleExtractor
from utils import HateSpeechDataset
from config import *
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def evaluate(model, dataloader, loss_bce, loss_ce):
    model.eval()
    total_loss = 0
    all_group_preds, all_group_labels = [], []
    all_hate_preds, all_hate_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target_start_labels = batch["target_start_labels"].to(DEVICE).float()
            target_end_labels = batch["target_end_labels"].to(DEVICE).float()
            arg_start_labels = batch["arg_start_labels"].to(DEVICE).float()
            arg_end_labels = batch["arg_end_labels"].to(DEVICE).float()
            group_labels = batch["group_labels"].to(DEVICE)
            hateful_labels = batch["hateful_labels"].to(DEVICE).long().squeeze()

            outputs = model(input_ids, attention_mask)
            target_start_logits, target_end_logits, arg_start_logits, arg_end_logits, group_logits, hateful_logits = outputs

            target_start_loss = loss_bce(target_start_logits, target_start_labels)
            target_end_loss = loss_bce(target_end_logits, target_end_labels)
            arg_start_loss = loss_bce(arg_start_logits, arg_start_labels)
            arg_end_loss = loss_bce(arg_end_logits, arg_end_labels)
            group_loss = loss_bce(group_logits, group_labels)
            hateful_loss = loss_ce(hateful_logits, hateful_labels)

            loss = target_start_loss + target_end_loss + arg_start_loss + arg_end_loss + group_loss + hateful_loss
            total_loss += loss.item()

            all_group_preds.append(torch.sigmoid(group_logits).cpu())
            all_group_labels.append(group_labels.cpu())
            all_hate_preds.append(torch.argmax(hateful_logits, dim=1).cpu())
            all_hate_labels.append(hateful_labels.cpu())

    group_f1 = f1_score(torch.cat(all_group_labels), (torch.cat(all_group_preds) > 0.5).float(), average="micro")
    hate_f1 = f1_score(torch.cat(all_hate_labels), torch.cat(all_hate_preds), average="binary")
    avg_loss = total_loss / len(dataloader)
    return avg_loss, group_f1, hate_f1


def train():
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)

    with open("data/train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    full_dataset = HateSpeechDataset("data/train.json", tokenizer)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = HateSpeechTupleExtractor(PRETRAINED_MODEL).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target_start_labels = batch["target_start_labels"].to(DEVICE).float()
            target_end_labels = batch["target_end_labels"].to(DEVICE).float()
            arg_start_labels = batch["arg_start_labels"].to(DEVICE).float()
            arg_end_labels = batch["arg_end_labels"].to(DEVICE).float()
            group_labels = batch["group_labels"].to(DEVICE)
            hateful_labels = batch["hateful_labels"].to(DEVICE).long().squeeze()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            target_start_logits, target_end_logits, arg_start_logits, arg_end_logits, group_logits, hateful_logits = outputs

            target_start_loss = loss_bce(target_start_logits, target_start_labels)
            target_end_loss = loss_bce(target_end_logits, target_end_labels)
            arg_start_loss = loss_bce(arg_start_logits, arg_start_labels)
            arg_end_loss = loss_bce(arg_end_logits, arg_end_labels)
            group_loss = loss_bce(group_logits, group_labels)
            hateful_loss = loss_ce(hateful_logits, hateful_labels)

            loss = target_start_loss + target_end_loss + arg_start_loss + arg_end_loss + group_loss + hateful_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss, val_group_f1, val_hate_f1 = evaluate(model, val_loader, loss_bce, loss_ce)
        print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Group F1={val_group_f1:.4f}, Hate F1={val_hate_f1:.4f}")

        # 保存最佳模型
        if val_group_f1 + val_hate_f1 > best_val_f1:
            best_val_f1 = val_group_f1 + val_hate_f1
            torch.save(model.state_dict(), "best_model.pt")
            print(f"✅ New best model saved at epoch {epoch+1}")

    print("Training complete.")


if __name__ == "__main__":
    train()