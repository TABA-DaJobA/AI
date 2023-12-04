import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from data.utils import get_data_path, get_folder_path
from data.info import DataPath, DataName, TrainType, FileFormat

def load_data(csv_file):
    return pd.read_csv(csv_file)

def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_score = torch.nn.functional.cosine_similarity(outputs[0], outputs[1], dim=1).item()
    return similarity_score

def create_eval_dataset(myself_csv, job_csv, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    myself_data = load_data(myself_csv)
    job_data = load_data(job_csv)

    eval_examples = []
    for intro, job_post in tqdm(zip(myself_data["sentence"], job_data["sentence"]), total=len(myself_data)):
        similarity_score = calculate_similarity(model, tokenizer, intro, job_post)
        eval_examples.append({"sentence1": intro, "sentence2": job_post, "score": similarity_score})

    return pd.DataFrame(eval_examples)

def main():
    dev_folder_path = get_folder_path(DataPath.ROOT, DataPath.DEV)
    eval_dataset = create_eval_dataset("data/train/myself_train.csv", "data/train/job_train.csv", "model_name")
    eval_save_path = get_data_path(
        folder_path=dev_folder_path,
        data_source=DataName.PREPROCESS_MYSELF,  # 예시로 사용, 실제 상황에 맞게 수정 가능
        train_type=TrainType.DEV,
        file_format=FileFormat.CSV
    )
    eval_dataset.to_csv(eval_save_path, index=False)
    print(f"Eval dataset saved to {eval_save_path}")

if __name__ == "__main__":
    main()
