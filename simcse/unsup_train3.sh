base="klue/"
name="roberta-small"
model_name="$base$name"
train_batch_size=256
step_num=10
OMP_NUM_THREADS=8

python train.py \
    --output_dir output/$name \
    --model_name_or_path $model_name \
    --train_file data/datasets/train \
    --pooler_type cls \
    --mlp_only_train \
    --temp 0.05 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $train_batch_size \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --do_train \
    --save_total_limit 6 \
    --logging_steps $step_num \
    --save_steps $step_num


