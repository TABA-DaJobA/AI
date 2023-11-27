import logging
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import is_main_process
from SimCSE.models import RobertaForCL, BertForCL
from SimCSE.arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from SimCSE.data_collator import SimCseDataCollatorWithPadding
from SimCSE.trainers import CLTrainer
import torch.distributed as dist

logger = logging.getLogger(__name__)

# 변수 설정
base = "klue/"
name = "roberta-small"
model_name = f"{base}{name}"
train_batch_size = 256
step_num = 10
OMP_NUM_THREADS = 8
output_dir = "output/roberta-small"  # 출력 디렉토리 설정


def main():
    # 데이터 인자와 훈련 인자를 초기화합니다.
    data_args = DataTrainingArguments(
        train_file="data/datasets/train",
        dev_file=None,  # 검증 데이터 파일 경로를 None으로 설정
        test_file=None,  # 테스트 데이터 파일 경로를 None으로 설정
        pad_to_max_length=False,  # 이 값을 True로 설정하면 데이터 패딩이 적용됩니다.
    )

    training_args = OurTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        learning_rate=5e-5,
        do_train=True,
        save_total_limit=6,
        logging_steps=step_num,
        save_steps=step_num,
        do_eval=True,
        load_best_model_at_end=True,
        evaluation_strategy="steps",  # 평가 전략을 steps로 변경
    )

    # ... (이후 코드는 그대로 유지)

    config = AutoConfig.from_pretrained(model_name)
    if "roberta" in model_name:
        model = RobertaForCL.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config,
            cache_dir=None,
            revision=None,
            use_auth_token=None,
            model_args=ModelArguments(do_mlm=False),  # 'do_mlm' 속성 설정
        )
    elif "bert" in model_name:
        model = BertForCL.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config,
            cache_dir=None,
            revision=None,
            use_auth_token=None,
            model_args=ModelArguments(do_mlm=False),  # 'do_mlm' 속성 설정
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_from_disk("data/datasets/train")

    # 검증 및 테스트 데이터가 없는 경우 빈 데이터셋을 생성
    dev_dataset = None
    test_dataset = None

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else SimCseDataCollatorWithPadding(
            tokenizer=tokenizer, data_args=data_args, model_args=ModelArguments(do_mlm=False)  # 'do_mlm' 속성 설정
        )
    )

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    if training_args.do_eval:
        if dev_dataset is not None:
            eval_result_on_valid_set = trainer.evaluate(dev_dataset)
            logger.info(
                f"Evaluation Result on the valid set! #####\n{eval_result_on_valid_set}"
            )
        if test_dataset is not None:
            eval_result_on_test_set = trainer.evaluate(test_dataset)
            logger.info(
                f"Evaluation Result on the test set! #####\n{eval_result_on_test_set}"
            )
        model.save_pretrained(f"{output_dir}/best_model")
        tokenizer.save_pretrained(f"{output_dir}/best_model")


if __name__ == "__main__":
    main()
