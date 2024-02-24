import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import random
import os
from typing import List, Dict, Tuple, Optional, NoReturn, Union
import json
import pandas as pd
import pickle
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import (
    DataLoader, RandomSampler, TensorDataset
)

from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    TrainingArguments
)

SEED = 180
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:

    def __init__(self,
                 args,
                 dataset,
                 sample_size,
                 num_neg,
                 tokenizer,
                 p_encoder,
                 q_encoder, 
                 model_id,
                 device,
                 model_save_dir = "../models/") -> None:
        
        self.args = args
        self.dataset = dataset
        self.sample_size = sample_size
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.model_id = model_id
        self.device = device
        self.model_save_dir = model_save_dir

        self.p_embeds = None

        self.prepare_passage_dataloader()
    
    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):
        print(f"Prepare in-batch negative samples with {num_neg} negative samples for each positive sample...")

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        sample_indices = np.random.choice(len(dataset), self.sample_size, replace=False)
        dataset = dataset.select(sample_indices)

        corpus = np.array(list(set([example for example in dataset["context"]]))) # remove duplicated contexts
        p_with_neg = []

        for c in tqdm(dataset["context"], desc="Batch negavtive sampling"):
            
            while True:
                neg_idxs = np.random.choice(len(corpus), size=self.num_neg)

                if c not in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        print(f"Negative sampling finished. Tokenization start...")
        q_seqs = tokenizer(dataset["question"], padding="max_length", truncation=True, return_tensors="pt", max_length=512)
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors="pt", max_length=512)
        print(f"Tokenization finished. Prepare dataloader...")

        max_len = p_seqs["input_ids"].size(-1) # max sequence length
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.train_batch_size)
        print(f"Train dataloader prepared.")
    

    def prepare_passage_dataloader(self):
        data_path = "../data/"
        context_path = "wikipedia_documents.json"

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(set([v["text"] for v in wiki.values()]))
        contexts_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors="pt", max_length=512)
        passage_dataset = TensorDataset(
            contexts_seqs['input_ids'], contexts_seqs['attention_mask'], contexts_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, shuffle=False, batch_size=self.args.eval_batch_size)
        print(f"Passage dataloader prepared.")
    

    def prepare_passage_embedding(self, p_encoder=None):

        if p_encoder is None:
            p_encoder = self.p_encoder
        
        p_encoder.eval()
        with torch.inference_mode():
            p_embeds = []
            for batch in tqdm(self.passage_dataloader, desc="Passage encoding"):
                batch = tuple(t.to(self.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                p_embeds.append(p_encoder(**p_inputs).last_hidden_state[:,0,:].detach().cpu())
            p_embeds = torch.cat(p_embeds, dim=0).view(len(self.passage_dataloader.dataset), -1)
        
        self.p_embeds = p_embeds
        print(f"Passage embedding prepared.")

    
    def train(self, args=None):

        if args is None:
            args = self.args
            
        batch_size = args.per_device_train_batch_size

        model_save_name = "_".join(self.model_id.split("/"))

        q_encoder_path = os.path.join(self.model_save_dir, f"{model_save_name}_q_encoder")
        p_encoder_path = os.path.join(self.model_save_dir, f"{model_save_name}_p_encoder")
        if os.path.exists(q_encoder_path) and os.path.exists(p_encoder_path):
            print(f"Pre-trained model exists. Load encoders from {self.model_save_dir}")
            self.q_encoder = AutoModel.from_pretrained(q_encoder_path).to(self.device)
            self.p_encoder = AutoModel.from_pretrained(p_encoder_path).to(self.device)
            self.prepare_passage_embedding()
            return
        
        print(f"Pre-trained model does not exist. Start training passage encoder and question encoder...")
        self.prepare_in_batch_negative()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for i in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long() # 각 데이터 모두 index 0 이 정답 -> [0, 0, 0, ... 0, 0, 0]
                    targets = targets.to(self.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(self.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(self.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(self.device)
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(self.device),
                        "attention_mask": batch[4].to(self.device),
                        "token_type_ids": batch[5].to(self.device)
                    }

                    p_outputs = self.p_encoder(**p_inputs).last_hidden_state[:,0,:]
                    q_outputs = self.q_encoder(**q_inputs).last_hidden_state[:,0,:]

                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(p_outputs, q_outputs.transpose(1, 2)).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                
                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}", step=f"{global_step+1}/{t_total}", lr=f"{scheduler.get_last_lr()[0]:.8f}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs
        
        print(f"Training finished.") 
        print(f"Save passage encoder under {p_encoder_path}...")
        self.p_encoder.save_pretrained(p_encoder_path)
        print(f"Save question encoder under {q_encoder_path}...")
        self.q_encoder.save_pretrained(q_encoder_path)
        print(f"Encoders saved.")

        self.prepare_passage_embedding()
        

    def retrieve(self, query_or_dataset, topk=1, args=None, p_encoder=None, q_encoder=None):
            
            if args is None:
                args = self.args
            
            if p_encoder is None:
                p_encoder = self.p_encoder
    
            if q_encoder is None:
                q_encoder = self.q_encoder
            
            if isinstance(query_or_dataset, str):
                doc_score, doc_index = self.get_relevant_doc(query_or_dataset, k=topk, args=args, p_encoder=p_encoder, q_encoder=q_encoder)
                print("[Search query]\n", query_or_dataset, "\n")

                for i in range(topk):
                    print(f"Top-{i+1} passage with score {doc_score[i]:4f}")
                    print(self.contexts[doc_index[i]])
                
                return (doc_score, [self.contexts[doc_index[i]] for i in range(topk)])

            elif isinstance(query_or_dataset, Dataset):
                
                total = []
                with timer("query exhaustive search"):
                    doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk, args=args, p_encoder=p_encoder, q_encoder=q_encoder)
                for i, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                    tmp = {
                        "question": example["question"],
                        "id": example["id"],
                        "context": " ".join([self.contexts[pid] for pid in doc_indices[i]])
                    }
                    if "context" in example.keys() and "answers" in example.keys():
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    total.append(tmp)
                
                cqas = pd.DataFrame(total)
                return cqas
    
    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):
        
        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder
        
        q_encoder.eval()
        with torch.inference_mode():
            q_seq = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt", max_length=512).to(self.device)
            q_embed = q_encoder(**q_seq).last_hidden_state[:,0,:].detach().cpu()

        dot_prod_scores = torch.matmul(q_embed, self.p_embeds.transpose(0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        doc_score = dot_prod_scores.squeeze()[rank].tolist()[:k]
        doc_index = rank.tolist()[:k]
        return doc_score, doc_index

    def get_relevant_doc_bulk(self, queries: List[str], k=1, args=None, p_encoder=None, q_encoder=None):
        
        if p_encoder is None:
            p_encoder = self.p_encoder
        
        if q_encoder is None:
            q_encoder = self.q_encoder

        p_encoder.eval()
        q_encoder.eval()
        with torch.inference_mode():
            q_embeds = []
            for q in tqdm(queries, desc="Query encoding"):
                q_seqs = self.tokenizer(q, padding="max_length", truncation=True, return_tensors="pt", max_length=512).to(self.device)
                q_embeds.append(q_encoder(**q_seqs).last_hidden_state[:,0,:].detach().cpu())
            q_embeds = torch.cat(q_embeds, dim=0)

        dot_prod_scores = torch.matmul(q_embeds, self.p_embeds.transpose(0, 1))

        doc_scores = []
        doc_indices = []
        for i in range(dot_prod_scores.size(0)):
            score, index = torch.topk(dot_prod_scores[i], k, largest=True, sorted=True)
            doc_scores.append(score.tolist())
            doc_indices.append(index.tolist())
        return doc_scores, doc_indices
    

if __name__ == "__main__":

    import argparse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current device: {device}")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        type=str,
        help="",
    )

    fargs = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(fargs.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(fargs.model_name_or_path, use_fast=False,)

    print("*" * 40, "Dense retrieval", "*" * 40)
    squad_dataset = load_dataset("squad_kor_v1")
    squad_dataset = squad_dataset["train"]

    p_encoder = AutoModel.from_pretrained(fargs.model_name_or_path).to(device)
    q_encoder = AutoModel.from_pretrained(fargs.model_name_or_path).to(device)

    targs = TrainingArguments(
        output_dir="../models",
        save_strategy="no",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    retriever = DenseRetrieval(
        args=targs,
        dataset=squad_dataset,
        sample_size=1000,
        num_neg=2,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        model_id=fargs.model_name_or_path,
        device=device,
    )

    retriever.train()


    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )

    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query)