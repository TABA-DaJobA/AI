base=klue/
name=roberta-base
model_name=$base$name

python make_datasets.py \
    --model_name_or_path $model_name \
    --train_file data/train/add_train.csv \
    --save_dir data/datasets

