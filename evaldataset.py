import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import random


def load_data(file_path):
    data = pd.read_csv(file_path)
    sentences = data['sentence'].tolist()
    return sentences


def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_score = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
    return similarity_score


def create_eval_dataset(num_samples, model_path, job_sentences, myself_sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    eval_examples = []
    used_indexes = []  # 이미 선택된 문장의 인덱스를 추적하기 위한 리스트

    for _ in tqdm(range(num_samples)):
        # 랜덤하게 job_sentences에서 문장 선택
        text1_index = random.randint(0, len(job_sentences) - 1)
        text1 = job_sentences[text1_index]

        # 랜덤하게 myself_sentences에서 문장 선택, 중복을 피하기 위해 이미 선택된 인덱스는 제외
        text2_index = random.randint(0, len(myself_sentences) - 1)
        while text2_index in used_indexes:
            text2_index = random.randint(0, len(myself_sentences) - 1)

        text2 = myself_sentences[text2_index]

        # 이미 선택된 인덱스를 추가하여 중복 선택 방지
        used_indexes.append(text2_index)

        similarity_score = calculate_similarity(model, tokenizer, text1, text2)
        eval_examples.append({"sentence1": text1, "sentence2": text2, "score": similarity_score})

    return pd.DataFrame(eval_examples)


def main():
    num_samples = 250  # 생성할 샘플 수를 지정하세요.
    model_path = "output/roberta-small/best_model"  # 여기에 모델 경로를 지정하세요.

    # job_train.csv와 myself_train.csv에서 문장 데이터를 가져옵니다.
    job_sentences = load_data("data/train/job_train.csv")
    myself_sentences = load_data("data/train/myself_train.csv")

    eval_dataset = create_eval_dataset(num_samples, model_path, job_sentences, myself_sentences)

    eval_dataset.to_csv("eval_dataset.csv", index=False)
    print(f"Eval dataset saved to eval_dataset.csv")

if __name__ == "__main__":
    main()

