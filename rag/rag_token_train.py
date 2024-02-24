import argparse
import transformers
import torch
import random
from tqdm.auto import tqdm
from datasets import DatasetDict, load_from_disk, load_metric
import faiss
import json
from sklearn.preprocessing import OneHotEncoder

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import logging

# 로그 파일 경로와 파일명
log_file_path = 'log.txt'

# 로거 생성
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 파일 핸들러 생성
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, contexts, targets=[], target_labels=[]):
        self.inputs = inputs
        self.contexts = contexts
        self.targets = targets
        self.target_labels = target_labels

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"],
                    "context": self.contexts[idx]}
        else:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"],
                    "context": self.contexts[idx],
                    'target_input_ids': self.targets[idx]["input_ids"],
                    'target_attention_mask': self.targets[idx]["attention_mask"],
                    'target_labels': self.target_labels[idx]["input_ids"]
                    }

    def __len__(self):
        return len(self.inputs)
    
class Dataloader(pl.LightningDataModule):
    def __init__(self, q_model_name, gen_model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.q_model_name = q_model_name
        self.gen_model_name = gen_model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.query_tokenizer = transformers.AutoTokenizer.from_pretrained(q_model_name)
        self.generation_tokenizer = transformers.AutoTokenizer.from_pretrained(gen_model_name)
        

    def tokenizing(self, data):
        input_data = []
        context_data = []
        target_data = []
        target_label_data = []

        for item in tqdm(data, desc='tokenizing', total=len(data)):
            question = item['question']
            context_data.append(item['question']) #1

            q_outputs = self.query_tokenizer(question, add_special_tokens=True, padding='max_length', truncation=True)
            for key in q_outputs:
                q_outputs[key] = torch.tensor(q_outputs[key], dtype=torch.long)

            try:
                answer = '<s>'+item['answers']['text'][0]+'</s>'
                target_answer = item['answers']['text'][0]+'</s>'
                a_outputs = self.generation_tokenizer(answer, max_length=30, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=False)
                a_target_outputs = self.generation_tokenizer(target_answer, max_length=30, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=False)
                
                for key in a_outputs.keys():
                    a_outputs[key] = torch.tensor(a_outputs[key], dtype=torch.long)
                    target_data.append(a_outputs) #2

                    a_target_outputs[key] = torch.tensor(a_target_outputs[key], dtype=torch.long)
                    target_label_data.append(a_target_outputs) #3
            except:
                pass

            input_data.append(q_outputs) #4

        return input_data, context_data, target_data, target_label_data

    def preprocessing(self, data):
        inputs, contexts, targets, labels = self.tokenizing(data)
        return inputs, contexts, targets, labels

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            datasets = load_from_disk(self.train_path)
            train_datasets, val_datasets = datasets['train'], datasets['validation']

            # 학습데이터 준비
            train_inputs, train_contexts, train_targets, train_labels = self.preprocessing(train_datasets)

            # 검증데이터 준비
            val_inputs, val_contexts ,val_targets, val_labels = self.preprocessing(val_datasets)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_contexts, train_targets, train_labels)
            self.val_dataset = Dataset(val_inputs, val_contexts, val_targets, val_labels)
        else:
            # 평가데이터 준비
            test_data = load_from_disk(self.test_path)
            test_inputs, test_contexts, test_targets, test_labels = self.preprocessing(test_data['validation'])
            self.test_dataset = Dataset(test_inputs, test_contexts, test_targets, test_labels)

            predict_data = load_from_disk(self.predict_path)
            predict_inputs, predict_contexts, _ , _  = self.preprocessing(predict_data['validation'])
            self.predict_dataset = Dataset(predict_inputs, predict_contexts)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

'''
def makeonehot(I):
    # I: (batch_size, num_retrieval)
    initial_one_hot = OneHotEncoder(categories=[range(56737)], sparse_output=False).fit_transform(I.reshape(-1, 1)) # (batch_size*num_retrieval, ntotla=56737)
    batch_one_hot = initial_one_hot.reshape(I.shape[0], I.shape[1], -1) # (batch_size, num_retrieval, ntotal=56737)
    return torch.tensor(batch_one_hot)
'''

def retrieval(q, index, wiki_doc_db, r_num=10):
    q_ = q.detach().cpu().numpy()
    score, relevant_wiki_id = index.search(q_, r_num) # (batch_size, num_retrieval = r_num), (batch_size, num_retrieval = r_num)
    batch_retrieved_docs = []
    for b in range(relevant_wiki_id.shape[0]):
        retrieved_docs = [wiki_doc_db[str(id)] for id in relevant_wiki_id[b]] # ( num_retrieval by *doc_length) : list
        batch_retrieved_docs.append(retrieved_docs)

    return score, batch_retrieved_docs # (batch_size, num_retrieval) : list, dtype:int / (batch_size, num_retrieval by *doc_length) : list, dtype:str

def concat_and_tokenize(context, batch_retrieved_docs, tokenizer):
    '''
    context : (batch_size by *context_length) : list
    batch_retrieved_docs : (batch_size, num_retrieval by *doc_length) : list, dtype:str
    '''
    # concat context and retrieved_docs by batch and tokenizing
    tokenized_context_docs = []
    for c, docs in zip(context, batch_retrieved_docs):
        '''
        c : (*context_length) : list
        docs : (num_retrieval by *doc_length) : list
        '''
        c_ = [c]*len(docs) # (num_retrieval by *context_length) : list

        tokenized = tokenizer(c_, docs, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_token_type_ids=False)
        for key in tokenized:
            tokenized[key] = torch.tensor(tokenized[key], dtype=torch.long).to('cuda')
        tokenized_context_docs.append(tokenized)
        
    return tokenized_context_docs # (batch_size, num_retrieval by dict[consists of 'input_ids', 'attention_mask']) : list


class TrainModel(pl.LightningModule):
    def __init__(self, q_model_name, gen_model_name , wiki_doc_db, gen_tokenizer ,lr, r_num=10):
        super().__init__()
        self.save_hyperparameters()

        self.q_model_name = q_model_name
        self.gen_model_name = gen_model_name
        self.gen_tokenizer = gen_tokenizer
        self.lr = lr
        self.r_num = r_num

        self.index = faiss.read_index('sent_emb.index') # ntotal wikidata, dim : 768
        # self.wiki_db = torch.load('index_to_vector.pt') # (ntotal, emb_dim=768)
        self.wiki_doc_db = wiki_doc_db # 

        self.query_encoder = transformers.AutoModel.from_pretrained(q_model_name, cache_dir='./tmp').to("cuda")
        self.encoder_decoder_generation_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(gen_model_name, cache_dir='./tmp').to("cuda")
        self.softmax = torch.nn.Softmax(dim=1)

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.encoder_decoder_generation_model.config.pad_token_id, reduction='mean')
    
    def forward(self, **x):
        '''
        x['input_ids'] : (batch_size, max_length),
        x['attention_mask'] : (batch_size, max_length)
        x['context'] : (batch_size) : list
        x['target_input_ids'] : (batch_size, answer_max_length),
        x['target_attention_mask'] : (batch_size, answer_max_length)
        x['target_labels'] : (batch_size, answer_max_length)
        '''

        q = self.query_encoder(**{'input_ids':x['input_ids'].to('cuda'), 'attention_mask':x['attention_mask'].to('cuda')})[1] # pooler_output : (batch_size, 768)
        
        # retrieval
        score, retrieved_docs = retrieval(q, self.index, self.wiki_doc_db, self.r_num) 

        # print('retrieved_docs : \n',len(retrieved_docs))
        # query-document concat and tokenizing
        gen_encoder_inputs = concat_and_tokenize(x['context'], retrieved_docs, self.gen_tokenizer) # (batch_size, num_retrieval by dict[consists of 'input_ids', 'attention_mask']) : list
        
        # generation per batch
        outputs = []
        for b in range(x['input_ids'].size(0)):
            # gen_encoder_inputs[b]['input_ids'].size() : (num_retrieval, max_length)
            # gen_encoder_inputs[b]['attention_mask'] : (num_retrieval, max_length)
            # x['target_input_ids'][b].size() : (answer_max_length) -> unsqueeze(0).expand(self.r_num,-1) -> (num_retrieval, answer_max_length)
            # x['target_attention_mask'][b].size() : (answer_max_length) -> unsqueeze(0).expand(self.r_num,-1) ->(num_retrieval, answer_max_length)

            output = self.encoder_decoder_generation_model(gen_encoder_inputs[b]['input_ids'],
                                                            attention_mask=gen_encoder_inputs[b]['attention_mask']
                                                           ,decoder_input_ids=x['target_input_ids'][b].unsqueeze(0).expand(self.r_num,-1).to('cuda'), 
                                                           decoder_attention_mask=x['target_attention_mask'][b].unsqueeze(0).expand(self.r_num,-1).to('cuda'))
            outputs.append(output.logits) # (num_retrieval, answer_max_length, vocab_size)
        outputs = torch.stack(outputs, dim=0) # (batch_size, num_retrieval, answer_max_length, vocab_size)
        
        # weighted mean outputs with score
        scores = self.softmax(torch.tensor(score)[:,:,None,None]) # (batch_size, num_retrieval, 1, 1)
        # (batch_size, num_retrieval, answer_max_length, vocab_size) -> (batch_size, answer_max_length, vocab_size)
        weighted_sum_outputs = torch.sum(scores*outputs.detach().cpu(), dim=1) 

        return weighted_sum_outputs.to('cuda') # (batch_size, answer_max_length, vocab_size)

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch['target_labels'] # (batch_size, answer_max_length)

        logits = self(**x)
        
        logits = logits.swapaxes(1, 2) # (batch_size, vocab_size, answer_max_length)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss.item())

        return loss
    
    def on_validation_epoch_start(self):
        self.em = 0
        self.count = 0

    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch['target_labels'] # (batch_size, answer_max_length)

        logits = self(**x)
        
        logits = logits.swapaxes(1, 2) # (batch_size, vocab_size, answer_max_length)
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss)

        # em metric
        predicts = torch.argmax(logits, dim=-2)
        for s, pred, y in zip(x['target_labels'], predicts,y):
            try:
                logger.debug(f'x: {s},\npred : {pred},\ny : {y}')
                pred_end = pred.tolist().index(1) # </s> token idx : 1
                y_end = y.tolist().index(1)
                # logger.debug(f'pred : {pred[:pred_end]}, y : {y[:y_end]}')
                if all(pred[:pred_end] == y[:y_end]):
                    self.em += 1
                else:
                    pass
            except:
                pass
        self.count += batch['input_ids'].size(0)

        return loss
    
    
    def on_validation_epoch_end(self):
        self.log("val_em", self.em/self.count)
        logger.debug(f'em : {self.em/self.count}')

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    

if __name__ == '__main__':

    config = {"q_model_name":'klue/bert-base',
              "gen_model_name":'gogamza/kobart-base-v2',
              "model_detail" : "v2",

              "batch_size": 8, 
              "shuffle":True,
              "learning_rate":1e-5,
              "rm_num":10,
              "epoch": 10,

              "train_path":'./data/train_dataset', 
              "dev_path":'./data/train_dataset',
              "test_path":'./data/train_dataset', 
              "predict_path":'./data/test_dataset',
              }

    wandb_logger = WandbLogger(project='baseline', entity='gypsi12')

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(config["q_model_name"], config["gen_model_name"],config["batch_size"],
                            config["shuffle"], config["train_path"], config["dev_path"],
                            config["test_path"], config["predict_path"])
    
    with open('unique_wiki_passages.json', 'r') as f:
        wiki_doc_db = json.load(f)
    
    model = TrainModel(config["q_model_name"], config["gen_model_name"], wiki_doc_db, dataloader.generation_tokenizer, config["learning_rate"], config["rm_num"])

    early_stop_custom_callback = EarlyStopping(
        "val_loss", patience=3, verbose=True, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        dirpath="./",
        filename='_'.join(config["q_model_name"].split()+config["gen_model_name"].split() + config["model_detail"].split()), # model에 따라 변화
        save_weights_only=False,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=config["epoch"], callbacks=[checkpoint_callback,early_stop_custom_callback],log_every_n_steps=1,logger=wandb_logger)

    # 학습
    trainer.fit(model=model, datamodule=dataloader)

    model = TrainModel(config["q_model_name"], config["gen_model_name"], config["learning_rate"], config["rm_num"])
    filename='_'.join(config["model_name"].split()+config["gen_model_name"].split() + config["model_detail"].split())
    model.load_from_checkpoint(f"{filename}.ckpt")

    # 저장
    torch.save(model, '_'.join(config["model_name"].split()+config["gen_model_name"].split() + config["model_detail"].split()) + '.pt')
