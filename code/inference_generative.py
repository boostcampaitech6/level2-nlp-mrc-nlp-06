"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys, os
import json
from typing import Callable, Dict, List, NoReturn, Tuple

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
from utils_qa_generative import check_no_error
import nltk
nltk.download('punkt')

logger = logging.getLogger(__name__)

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

    training_args.do_predict = True
    training_args.do_eval = False
    training_args.do_train = False

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

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
    print(datasets)

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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=CACHE_PATH
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args,
        )

    print('## training_args', training_args)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "/data/ephemeral/level2-nlp-mrc-nlp-06/data/",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
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
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
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

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
        
        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # evaluation인 경우 target이 존재함
        if training_args.do_eval:
            targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
            labels = tokenizer(
                text_target=targets,
                truncation=True,
                max_length=data_args.max_answer_length,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            model_inputs["labels"] = labels["input_ids"]


        model_inputs["example_id"] = []
        print('## shape check : ', len(model_inputs), len(model_inputs["input_ids"]))

        for i in range(len(model_inputs["input_ids"])):
            model_inputs["example_id"].append(examples["id"][i])

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

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    metric = load_metric("squad")

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
        # 여기서 do_predict, do_eval에 따라 처리 다르게!
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print(f'decoded preds: {decoded_preds}, labels: {decoded_labels}')
        
        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]

        # 예측값 반환
        if training_args.do_predict:
            return formatted_predictions
        
        # eval 후 metric 반환
        elif training_args.do_eval:
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
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

    logger.info("*** Evaluate ***")

    def generate_answer(encode_sentence):
        pred = tokenizer.decode(encode_sentence, skip_special_tokens=True)
        pred = pred.replace("\n", "") # 줄바꿈 제거
        pred = pred.replace("\\n", "") # 줄바꿈 제거
        pred = pred.replace("context", "") # context 제거
        pred = pred.replace(":", "") # 문장부호 제거
        pred = pred.replace(".", "") # 문장부호 제거
        pred = pred.strip() # 좌우 공백 제거
        return pred

    # prediction 진행
    if training_args.do_predict:
        encoded_predictions = trainer.predict(
            test_dataset=eval_dataset
        )
        # print('### encoded predictions : ', encoded_predictions)
        # print('### predictions shape', encoded_predictions.predictions)
        # print('### eval_dataset : ', eval_dataset)
        # print('### eval_dataset id : ', eval_dataset['example_id'])

        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = collections.OrderedDict()

        # sentence에 대한 decoding 진행
        for idx, sample_id in enumerate(tqdm(eval_dataset['example_id'], desc="decoding")):
            print('## sample id : ', sample_id)
            all_predictions[sample_id] = generate_answer(encoded_predictions.predictions[idx])

        print('### decoded predictions : ', all_predictions)      

        # prediction 저장
        # output_dir이 있으면 모든 dicts를 저장합니다.
        if training_args.output_dir is not None:
            assert os.path.isdir(training_args.output_dir), f"{training_args.output_dir} is not a directory."

            prediction_file = os.path.join(
                training_args.output_dir,
                "predictions.json" if model_args.prefix is None else f"predictions_{model_args.prefix}".json,
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
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    


if __name__ == "__main__":
    main()
