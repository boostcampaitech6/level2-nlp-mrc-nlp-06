"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys, os
import json
from typing import Callable, Dict, List, NoReturn, Tuple
import wandb

import numpy as np
from arguments_generative import DataTrainingArguments, ModelArguments, CustomizedTrainingArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval
from retrieval_bm25 import BM25SparseRetrieval

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)
from tqdm import tqdm
import collections
from utils_qa import check_no_error

logger = logging.getLogger(__name__)

# wandb project 설정
os.environ["WANDB_ENTITY"] = "be-our-friend"
os.environ["WANDB_PROJECT"] = "MRC_HYW"


# cache path 추가
CACHE_PATH = "/data/ephemeral/level2-nlp-mrc-nlp-06/cache"
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomizedTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f'## WANDB RUN NAME : {training_args.run_name} ##')

    ## evaluation & predict 설정
    training_args.do_predict = False
    training_args.do_eval = True
    training_args.do_train = False

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    print('## model_args', model_args)
    print('## data_args', data_args)
    print('## training_args', training_args)

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
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
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=CACHE_PATH
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
        cache_dir=CACHE_PATH
    )
    # 체크포인트에서 불러오기
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=CACHE_PATH
    )

    print(
        model_args.model_name_or_path,
        type(model),
    )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args,
        )
    

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: CustomizedTrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "/data/ephemeral/level2-nlp-mrc-nlp-06/data/",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    ## retrieval 정하기
    if data_args.retrieval_model == 'tfidf':
        retriever = SparseRetrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
        print('## TF-IDF Retrieval model Selected ##')
    
    elif data_args.retrieval_model == 'bm25':
        retriever = BM25SparseRetrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
        print('## BM25 Retrieval model Selected ##')
        
    
    else:
        raise ValueError("Choose One of tfidf, bm25")


    retriever.get_sparse_embedding()
    
    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Value(dtype="string", id=None),
                "context": Value(dtype="string", id=None),
                "original_context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    def clean_text(sentence):
        sentence = sentence.replace("\\n", "") # 줄바꿈 제거
        sentence = sentence.replace("\n", "") # 줄바꿈 제거
        sentence = sentence.replace("\\", "") # 특수 기호
        sentence = sentence.replace("context", "") # context 제거
        sentence = sentence.replace(":", "") # 문장부호 제거
        sentence = sentence.replace(".", "") # 문장부호 제거
        sentence = sentence.replace("\"", "") # 문장부호 제거
        sentence = sentence.strip() # 좌우 공백 제거

        return sentence

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        print('### prepare validation features ###')
        
        # 탐색 완료한 examples 반환
        inputs = [f"question: {clean_text(q)}  context: {clean_text(c)} </s>" for q, c in zip(examples["question"], examples["context"])]
        print('## input example : ', inputs[:5])
        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # evaluation인 경우 target이 존재함
        if training_args.do_eval:
            targets = [f'{clean_text(a)} </s>' for a in examples['answers']]
            print('## target example : ', targets[:5])
            labels = tokenizer(
                text_target=targets,
                truncation=True,
                max_length=data_args.max_answer_length,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            model_inputs["labels"] = labels["input_ids"]

        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            model_inputs["example_id"].append(examples["id"][i])

        print('### model inputs : ', model_inputs.keys())
        
        return model_inputs


    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
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
    # nltk를 이용한 간단한 후처리 진행
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    # data input 바꿔줌
    metric = load_metric('/data/ephemeral/level2-nlp-mrc-nlp-06/code/squad_generative.py')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        
        # overflowerror 방지
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        # print(f'## encoded preds example : {tokenizer.decode(preds[0], skip_special_tokens=True)}')
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # print(f'## encoded labels example : {tokenizer.decode(labels[0], skip_special_tokens=True)}')
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # postprocess 진행
        # print(f"## before postprocess : decoded preds : {decoded_preds}, decoded labels : {decoded_labels}")
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # print(f"## after postprocess : decoded preds : {decoded_preds}, decoded labels : {decoded_labels}")

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]

        # 예측값 반환
        if training_args.do_predict:
            return formatted_predictions
        
        # eval 후 metric 반환
        elif training_args.do_eval:
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

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
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    def generate_answer(encode_sentence):
        pred = tokenizer.decode(encode_sentence, skip_special_tokens=True)
        pred = clean_text(pred)
        return pred


    # prediction 진행
    if training_args.do_predict:
        # 학습된 것을 바탕으로 predict 진행
        encoded_predictions = trainer.predict(
            test_dataset=eval_dataset
        )

        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = collections.OrderedDict()

        # sentence에 대한 decoding 진행
        for idx, sample_id in enumerate(tqdm(eval_dataset['example_id'], desc="decoding")):
            all_predictions[sample_id] = generate_answer(encoded_predictions.predictions[idx])
            # print(f'### {idx} ({sample_id}) 번째 prediction / answer ###')
            # print(f'## predicition : {all_predictions[sample_id]} || answer : {eval_dataset["answer"][idx]}')


        # prediction 저장
        # output_dir이 있으면 모든 dicts를 저장합니다.
        if training_args.output_dir is not None:
            assert os.path.isdir(training_args.output_dir), f"{training_args.output_dir} is not a directory."

            prediction_file = os.path.join(
                training_args.output_dir,
                "predictions.json" if model_args.prefix is None else f"predictions_{model_args.prefix}.json",
            )

            with open(prediction_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
                )

        print(
            f"prediction file is saved in {training_args.output_dir}!"
        )

    # evaluation 하기
    if training_args.do_eval:
        print("Evaluate 진행")
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        # evaluation에서 반환되는 결과값을 보자
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        wandb.log({"eval": metrics})



if __name__ == "__main__":
    main()