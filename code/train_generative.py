import logging
import os
import sys
import random
import numpy as np
import torch
from typing import NoReturn

from arguments_generative import DataTrainingArguments, ModelArguments, CustomizedTrainingArguments
from datasets import DatasetDict, load_from_disk, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

from utils_qa_generative import check_no_error

# wandb
import wandb
wandb.login()


# seed
seed = 2024
deterministic = False

random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)
if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__name__)

# cache path 추가
CACHE_PATH = "/data/ephemeral/level2-nlp-mrc-nlp-06/cache"
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# wandb project 설정
os.environ["WANDB_ENTITY"] = "be-our-friend"
os.environ["WANDB_PROJECT"] = "MRC_HYW"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # fp16 설정 가능
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomizedTrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)
    print(f'## WANDB RUN NAME : {training_args.run_name} ##')

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다


    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.


    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
        cache_dir=CACHE_PATH
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
        cache_dir=CACHE_PATH
    )
    # seq2seq 모델이므로 불러오기
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=CACHE_PATH
    )

    print('## training_args', training_args)
    print("## model args", model_args)
    print('## data args', data_args)

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: CustomizedTrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        train_column_names = datasets["train"].column_names
    if training_args.do_eval:
        valid_column_names = datasets["validation"].column_names

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    def clean_text(sentence):
        if type(sentence) == dict:
            print("## Sentence : ", sentence)
            sentence = sentence["text"][0]
            print("## Changed Sentence : ", sentence)
            
        sentence = sentence.replace("\\n", "") # 줄바꿈 제거
        sentence = sentence.replace("\n", "") # 줄바꿈 제거
        sentence = sentence.replace("\\", "") # 특수 기호
        sentence = sentence.replace("context", "") # context 제거
        sentence = sentence.replace(":", "") # 문장부호 제거
        sentence = sentence.replace(".", "") # 문장부호 제거
        sentence = sentence.replace("\"", "") # 문장부호 제거
        sentence = sentence.strip() # 좌우 공백 제거

        return sentence

    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):

        inputs = [f"question: {clean_text(q)}  context: {clean_text(c)} </s>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f"{clean_text(a)} </s>" for a in examples['answers']]

        print('### input samples : ', inputs[:5])
        print('### targets samples : ', targets[:5])

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        labels = tokenizer(
            text_target=targets,
            truncation=True,
            max_length=data_args.max_answer_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        
        print('### model inputs : ', model_inputs.keys())

        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_column_names,
            load_from_cache_file = not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.

        inputs = [f"question: {clean_text(q)}  context: {clean_text(c)} </s>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f'{clean_text(a)} </s>' for a in examples['answers']]

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        labels = tokenizer(
            text_target=targets,
            truncation=True,
            max_length=data_args.max_answer_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])

        print('### model inputs : ', model_inputs.keys())

        return model_inputs


    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=valid_column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # preds = ["\n".join(tokenizer(pred)) for pred in preds]
        # labels = ["\n".join(tokenizer(label)) for label in labels]

        return preds, labels


    metric = load_metric('/data/ephemeral/level2-nlp-mrc-nlp-06/code/squad_generative.py')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        
        # overflowerror 방지
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # postprocess 진행
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # print(f'## decoded preds : {decoded_preds}, decoded labels : {decoded_labels}')
        
        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]["text"][0]} for ex in datasets["validation"]]

        
        print("### reference and prediction check ###")
        for idx, id in enumerate(datasets["validation"]["id"]):
            print(f"### ID : {id} ###")
            print(f"### ref : {references[idx]}, pred : {formatted_predictions[idx]}")

        print('### Check finished! ###')
        # squad 바꿔줬음
        result = metric.compute(predictions=formatted_predictions, references=references)

        return {"eval_exact_match": result['exact_match'], "eval_f1": result['f1']}

    print("init trainer...")
    # Trainer 초기화
    # early stopping 추가
    # postprocess 따로 진행해줘야 함
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    logger.info("*** Train ***")
    
    # epoch 마다 validation 진행
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
                wandb.log({str(key): value})

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        print("## Evaluate 진행 ##")
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        print('## Evaluate metrics 기록 : ', metrics)

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        wandb.log({"eval": metrics})



if __name__ == "__main__":
    main()