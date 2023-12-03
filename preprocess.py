from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from data.info import (
    DataName,
    DataPath,
    STSDatasetFeatures,
    TCDatasetFeatures,
    TrainType,
    FileFormat,
    TrainType,
    UnsupervisedSimCseFeatures,
)
from data.utils import (
    get_data_path,
    get_folder_path,
    raw_data_to_dataframe,
    make_unsupervised_sentence_data,
    #wiki_preprocess,
    job_preprocess,
    change_col_name,
    add_sts_df,
)

import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
"""
채용공고_preprocess
"""
job_dataset = pd.read_csv("cop.csv")
##############
data = []
for job_text in tqdm(job_dataset):
    job_text = job_text.replace(". ", ".\n")
    job_text = job_text.replace("\xa0", " ")
    job_sentences = job_text.split("\n")

    for job_sentence in job_sentences:
        job_sentence = job_sentence.rstrip().lstrip()
        if len(job_sentence) >= 10:
            data.append(job_sentence)
# data/utils.py로 옮기기
###########################################################
job_df = pd.DataFrame(data={UnsupervisedSimCseFeatures.SENTENCE.value: data})
job_df[UnsupervisedSimCseFeatures.SENTENCE.value] = job_df[
    UnsupervisedSimCseFeatures.SENTENCE.value
].apply(job_preprocess)

"""
wiki_preprocess

wiki_dataset = load_dataset("sh110495/kor-wikipedia")
#######################################################
data = []
for wiki_text in tqdm(wiki_dataset["train"]["text"]):
    wiki_text = wiki_text.replace(". ", ".\n")
    wiki_text = wiki_text.replace("\xa0", " ")
    wiki_sentences = wiki_text.split("\n")

    for wiki_sentence in wiki_sentences:
        wiki_sentence = wiki_sentence.rstrip().lstrip()
        if len(wiki_sentence) >= 10:
            data.append(wiki_sentence)
# data/utils.py로 옮기기
###########################################################
wiki_df = pd.DataFrame(data={UnsupervisedSimCseFeatures.SENTENCE.value: data})
wiki_df[UnsupervisedSimCseFeatures.SENTENCE.value] = wiki_df[
    UnsupervisedSimCseFeatures.SENTENCE.value
].apply(wiki_preprocess)
"""

"""
sts_preprocess

"""
train_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.TRAIN)
dev_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.DEV)
test_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.TEST)
"""
preprocess_wiki_data_path = get_data_path(
    folder_path=train_floder_path,
    data_source=DataName.PREPROCESS_WIKI,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)
"""
preprocess_job_data_path = get_data_path(
    folder_path=train_floder_path,
    data_source=DataName.PREPROCESS_JOB,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)


job_df = job_df.dropna(axis=0)
job_df.to_csv(preprocess_job_data_path, index =False)
#wiki_df = wiki_df.dropna(axis=0)
#wiki_df.to_csv(preprocess_wiki_data_path, index=False)
# wiki_df = pd.read_csv(preprocess_wiki_data_path)
logging.info(
    f"preprocess job train done!\nfeatures:{job_df.columns} \nlen: {len(job_df)}\nna count:{sum(job_df[UnsupervisedSimCseFeatures.SENTENCE.value].isna())}"
)


cop_data = []
with open("cop.csv", "r", encoding="utf-8") as file:
    cop_data = [line.strip() for line in file if line.strip()]

cop_train = pd.DataFrame({
    'sentence1': cop_data,
    'sentence2': cop_data
})
##############

data = []
with open("cop.csv", "r", encoding="utf-8") as file:
    lines = file.read().split("\n")
    for cop_text in lines:
        # 이 부분은 이미 사용하고 있는 전처리 함수를 호출합니다.
        cop_text = job_preprocess(cop_text)
        # 길이가 충분히 큰 문장만 추가합니다.
        if len(cop_text) >= 10:
            data.append(cop_text)


# 'sentence' 열을 생성하고 job_preprocess 함수를 적용
cop_df = pd.DataFrame(data={'sentence': data})
cop_df['sentence'] = cop_df['sentence'].apply(job_preprocess)

# 'sentence1'과 'sentence2' 열을 추가
cop_df['sentence1'] = cop_df['sentence']
cop_df['sentence2'] = cop_df['sentence']

#크롤링 데이터
#cop_dev = cop_df

#cop_test = cop_df

"""
klue_train = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KLUE, TrainType.TRAIN, FileFormat.JSON
)

kakao_train = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KAKAO, TrainType.TRAIN, FileFormat.TSV
)

kakao_dev = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KAKAO, TrainType.DEV, FileFormat.TSV
)

kakao_test = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_KAKAO, TrainType.TEST, FileFormat.TSV
)
"""
"""
####
# tc
tc_train = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_TC, TrainType.TRAIN, FileFormat.JSON
)
tc_dev = raw_data_to_dataframe(
    DataPath.ROOT, DataPath.RAW, DataName.RAW_TC, TrainType.DEV, FileFormat.JSON
)

tc_train = change_col_name(
    tc_train, TCDatasetFeatures.TITLE, UnsupervisedSimCseFeatures.SENTENCE
)
tc_dev = change_col_name(
    tc_dev, TCDatasetFeatures.TITLE, UnsupervisedSimCseFeatures.SENTENCE
)
"""
#print(tc_dev.head())
print(job_df.head())
#print(wiki_df.head())
#
####
unsup_datas = [job_df] #데이터추가
total_sen = []
for unsup_data in unsup_datas:
    total_sen.extend(unsup_data[UnsupervisedSimCseFeatures.SENTENCE.value].to_list())
total_unsup_df = pd.DataFrame(
    data={UnsupervisedSimCseFeatures.SENTENCE.value: total_sen}
)

preprocess_add_data_path = get_data_path(
    folder_path=train_floder_path,
    data_source=DataName.PREPROCESS_ADD,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)
total_unsup_df.to_csv(preprocess_add_data_path, index=False)



sts_train_list = [cop_train] #문장유사성 비교데이터

train_sentence_list = make_unsupervised_sentence_data(sts_data_list=sts_train_list)

preprocess_sts_train_path = get_data_path(
    train_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)

preprocess_sts_dev_path = get_data_path(
    dev_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.DEV,
    file_format=FileFormat.CSV,
)

preprocess_sts_test_path = get_data_path(
    test_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.TEST,
    file_format=FileFormat.CSV,
)


sts_train_df = pd.DataFrame(
    data={UnsupervisedSimCseFeatures.SENTENCE.value: train_sentence_list}
)
sts_train_df.dropna(axis=0).to_csv(preprocess_sts_train_path, index=False)

logging.info(
    f"preprocess sts train done!\nfeatures:{sts_train_df.columns} \nlen: {len(sts_train_df)}"
)

# sts_dev_df = kakao_dev[
#     [
#         STSDatasetFeatures.SENTENCE1.value,
#         STSDatasetFeatures.SENTENCE2.value,
#         STSDatasetFeatures.SCORE.value,
#     ]
# ]
# sts_test_df = kakao_test[
#     [
#         STSDatasetFeatures.SENTENCE1.value,
#         STSDatasetFeatures.SENTENCE2.value,
#         STSDatasetFeatures.SCORE.value,
#     ]
# ]

# sts_dev_df.to_csv(preprocess_sts_dev_path, sep="\t", index=False)

# logging.info(
#     f"preprocess sts dev done!\nfeatures:{sts_dev_df.columns} \nlen: {len(sts_dev_df)}"
# )

# sts_test_df.to_csv(preprocess_sts_test_path, sep="\t", index=False)

# logging.info(
#     f"preprocess sts test done!\nfeatures:{sts_test_df.columns} \nlen: {len(sts_test_df)}"
# )
