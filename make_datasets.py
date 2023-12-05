import pandas as pd
import torch

from data.info import UnsupervisedSimCseFeatures, STSDatasetFeatures
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModel
from SimCSE.arguments import DataTrainingArguments
from data.info import UnsupervisedSimCseFeatures, STSDatasetFeatures

# 비감독전처리함수
def unsupervised_prepare_features(examples, tokenizer, data_args):
    total = len(examples[UnsupervisedSimCseFeatures.SENTENCE.value])
    # Avoid "None" fields
    for idx in range(total):
        if examples[UnsupervisedSimCseFeatures.SENTENCE.value][idx] is None:
            examples[UnsupervisedSimCseFeatures.SENTENCE.value][idx] = " "

    sentences = (
        examples[UnsupervisedSimCseFeatures.SENTENCE.value]
        + examples[UnsupervisedSimCseFeatures.SENTENCE.value]
    )

    sent_features = tokenizer(
        sentences,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding="max_length",
    )

    features = {}

    for key in sent_features:
        features[key] = [
            [sent_features[key][i], sent_features[key][i + total]] for i in range(total)
        ]
    return features

# 감독학습 전처리
def sts_prepare_features(examples, tokenizer, data_args):
    total = len(examples[STSDatasetFeatures.SENTENCE1.value])
    # Avoid "None" fields
    scores = []
    for idx in range(total):
        score = examples[STSDatasetFeatures.SCORE.value][idx]
        if score is None:
            score = 0
        if examples[STSDatasetFeatures.SENTENCE1.value][idx] is None:
            examples[STSDatasetFeatures.SENTENCE1.value][idx] = " "
        if examples[STSDatasetFeatures.SENTENCE2.value][idx] is None:
            examples[STSDatasetFeatures.SENTENCE2.value][idx] = " "
        scores.append(score / 5.0)

    sentences = (
        examples[STSDatasetFeatures.SENTENCE1.value]
        + examples[STSDatasetFeatures.SENTENCE2.value]
    )

    sent_features = tokenizer(
        sentences,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding="max_length",
    )

    features = {}

    for key in sent_features:
        features[key] = [
            [sent_features[key][i], sent_features[key][i + total]] for i in range(total)
        ]
    features["labels"] = scores

    return features


def main(model_name_or_path, train_file, dev_file, test_file, save_dir):
    data_args = DataTrainingArguments(
        train_file=train_file,
        dev_file=dev_file,
        test_file=test_file,
        save_dir=save_dir,
        preprocessing_num_workers=4,
        overwrite_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    train_data_files = {}
    eval_data_files = {}
    if train_file is not None:
        train_data_files["train"] = train_file
        print("train data success")
    else:
        # 훈련 데이터가 없으면 종료
        print("no train data")
        return

    if dev_file is not None:
        eval_data_files["dev"] = dev_file
        print("dev data success")
    else:
        # dev 데이터가 없으면 none 처리
        eval_data_files["dev"] = None
        print("no dev data")

    if test_file is not None:
        eval_data_files["test"] = test_file
        print("test data success")
    else:
        # test 데이터가 없으면 none 처리
        eval_data_files["test"] = None
        print("no test data")

    train_extension = train_file.split(".")[-1]
    valid_extension = "csv"

    train_dataset = load_dataset(
        train_extension,
        data_files=train_data_files,
        cache_dir=None,
    )

    print("Train dataset loaded successfully.")
    print(f"Train file path: {train_file}")

    valid_dataset = None
    if eval_data_files["dev"] is not None:
        valid_dataset = load_dataset(
            valid_extension,
            data_files=eval_data_files,
            cache_dir=None,
            delimiter="\t",
        )

    unsup_prepare_features_with_param = partial(
        unsupervised_prepare_features, tokenizer=tokenizer, data_args=data_args
    )
    dev_prepare_features_with_param = partial(
        sts_prepare_features, tokenizer=tokenizer, data_args=data_args
    )
    valid_column_names = valid_dataset["dev"].column_names if valid_dataset is not None else []

    train_dataset = (
        train_dataset["train"]
        .map(
            unsup_prepare_features_with_param,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=UnsupervisedSimCseFeatures.SENTENCE.value,
            load_from_cache_file=False,
        )
        .save_to_disk(data_args.save_dir + "/train")
    )

    if valid_dataset is not None:
        valid_dataset["dev"] = valid_dataset["dev"].map(
            dev_prepare_features_with_param,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=valid_column_names,
            load_from_cache_file=False,
        )
        valid_dataset["test"] = valid_dataset["test"].map(
            dev_prepare_features_with_param,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=valid_column_names,
        )

if __name__ == "__main__":
    model_name_or_path = "klue/roberta-base"
    train_file = "data/train/myself_train.csv"
    dev_file = None  # 'sts_dev.tsv' 파일이 없으므로 None으로 설정
    test_file = None  # 'sts_test.tsv' 파일이 없으므로 None으로 설정
    save_dir = "data/datasets1"

    main(model_name_or_path, train_file, dev_file, test_file, save_dir)
    print("Dataset processing completed successfully.")
