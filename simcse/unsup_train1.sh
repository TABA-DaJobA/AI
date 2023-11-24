base="klue/"
name="roberta-small"
model_name="$base$name"
train_batch_size=256
step_num=10


python make_datasets.py \
    --model_name_or_path $model_name \
    --train_file data/train/job_train.csv \
    --save_dir data/datasets \
&&
python train.py \
    --output_dir output/$name \
    --model_name_or_path $model_name \
    --train_file data/datasets/train \
    --dev_file data/datasets/validation \
    --pooler_type cls \
    --mlp_only_train \
    --temp 0.05 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --label_names labels \
    --per_device_train_batch_size $train_batch_size \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --do_train \
    --do_eval \
    --save_total_limit 6 \
    --logging_steps $step_num \
    --save_steps $step_num \
    --eval_steps $step_num \
    --load_best_model_at_end \
    --fp16 \
    --metric_for_best_model eval_cosine_spearman \
    --no_cuda
