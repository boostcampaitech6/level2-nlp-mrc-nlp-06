from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments
import os

OUTPUT_PATH = "/data/ephemeral/level2-nlp-mrc-nlp-06/outputs"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        # 실습에서 사용된 모델
        default="paust/pko-t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="/data/ephemeral/level2-nlp-mrc-nlp-06/data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )

class CustomizedTrainingArguments(Seq2SeqTrainingArguments):
    def __init__(self, output_dir=OUTPUT_PATH, *args, **kwargs):
        kwargs["do_train"] = True
        kwargs["do_eval"] = True
        kwargs["predict_with_generate"] = True
        kwargs["report_to"] = "wandb"
        kwargs["num_train_epochs"] = 30
        kwargs["logging_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"
        kwargs["save_total_limit"] = 2
        kwargs["evaluation_strategy"] = "epoch"
        kwargs["metric_for_best_model"] = "eval_exact_match"
        kwargs["greater_is_better"] = True

        kwargs["load_best_model_at_end"] = True

        # 부모 클래스의 __init__ 호출
        super().__init__(output_dir, *args, **kwargs)