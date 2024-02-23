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
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"]}
        else:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"],
                    'target_input_ids': self.targets[idx]["input_ids"],
                    'target_attention_mask': self.targets[idx]["attention_mask"]}

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
        target_data = []
        for item in tqdm(data, desc='tokenizing', total=len(data)):
            question = item['question']
            q_outputs = self.query_tokenizer(question, add_special_tokens=True, padding='max_length', truncation=True)

            try:
                answer = '<s>'+item['answers']['text'][0]+'</s>'
                a_outputs = self.generation_tokenizer(answer, max_length=30, add_special_tokens=True, padding='max_length', truncation=True)
                for key in q_outputs:
                    a_outputs[key] = torch.tensor(a_outputs[key], dtype=torch.long)
                    target_data.append(a_outputs)
            except:
                pass

            for key in q_outputs:
                q_outputs[key] = torch.tensor(q_outputs[key], dtype=torch.long)

            input_data.append(q_outputs)
            
        return input_data, target_data

    def preprocessing(self, data):
        inputs, targets = self.tokenizing(data)
        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            datasets = load_from_disk(self.train_path)
            train_datasets, val_datasets = datasets['train'], datasets['validation']

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_datasets)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_datasets)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = load_from_disk(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data['validation'])
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = load_from_disk(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data['validation'])
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

def makeonehot(I):
    # I: (batch_size, num_retrieval)
    initial_one_hot = OneHotEncoder(categories=[range(56737)], sparse_output=False).fit_transform(I.reshape(-1, 1)) # (batch_size*num_retrieval, ntotla=56737)
    batch_one_hot = initial_one_hot.reshape(I.shape[0], I.shape[1], -1) # (batch_size, num_retrieval, ntotal=56737)
    return torch.tensor(batch_one_hot)

class TrainModel(pl.LightningModule):
    def __init__(self, q_model_name, gen_model_name ,lr, r_num=10):
        super().__init__()
        self.save_hyperparameters()

        self.q_model_name = q_model_name
        self.gen_model_name = gen_model_name
        self.lr = lr
        self.r_num = r_num

        self.index = faiss.read_index('sent_emb.index') # ntotal : 56737 wikidata, dim : 768
        self.wiki_db = torch.load('index_to_vector.pt') # (ntotal=56737, emb_dim=768)

        self.query_encoder = transformers.AutoModel.from_pretrained(q_model_name, cache_dir='/data/ephemeral/tmp').to("cuda")
        self.generation_decoder = transformers.AutoModelForCausalLM.from_pretrained(gen_model_name, cache_dir='/data/ephemeral/tmp').to("cuda")
        self.resize_layer = torch.nn.Linear(768*2, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.norm = torch.nn.BatchNorm1d(num_features=768)

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.generation_decoder.config.pad_token_id, reduction='mean')
    
    def forward(self, **x):
        '''
        x['input_ids'] : (batch_size, max_length),
        x['attention_mask'] : (batch_size, max_length)
        x['target_input_ids'] : (batch_size, answer_max_length),
        x['target_attention_mask'] : (batch_size, answer_max_length)
        '''
        q = self.query_encoder(**{'input_ids':x['input_ids'], 'attention_mask':x['attention_mask']})[1].to('cuda') # pooler_output : (batch_size, 768)
        # retrieval
        q_ = q.detach().cpu().numpy()
        score, relevant_wiki_id = self.index.search(q_, self.r_num) # (batch_size, num_retrieval = r_num), (batch_size, num_retrieval = r_num)
        one_hot = makeonehot(relevant_wiki_id)  # (batch_size, num_retrieval, ntotal=56737)
        retrieved_vectors = torch.matmul(one_hot, self.wiki_db.double()) # (batch_size, num_retrieval, emb_dim=768)

        # query-document concat
        query_vectors = q.unsqueeze(1).expand(-1, self.r_num, -1).to('cpu') # (batch_size, num_retrieval, emb_dim=768)
        concat_vectors = torch.cat([query_vectors, retrieved_vectors], dim=-1) # (batch_size, num_retrieval, emb_dim*2=1536)

        # resize
        resized_vectors = self.resize_layer(concat_vectors.to(torch.float).to('cuda')) # (batch_size, num_retrieval, emb_dim=768)
        normalized_vectors = self.norm(resized_vectors.view(-1, 768)).view(resized_vectors.size())
        gen_start_vector = self.dropout(normalized_vectors) 

        # generation
        original_input_imbeds = self.generation_decoder.get_input_embeddings()(x['target_input_ids']) # (batch_size, answer_max_length, emb_dim=768)
        # starting from gen_start_vector
        # (batch_size, num_retrieval, 1, emb_dim) [concat] (batch_size, num_retrieval, answer_max_length-1, emb_dim=768) -> (batch_size, num_retrieval, answer_max_length, emb_dim=768)
        input_imbeds = torch.cat([gen_start_vector.unsqueeze(-2), original_input_imbeds.unsqueeze(1).expand(-1,self.r_num,-1,-1)[:,:,:-1,:]], dim=-2)
        
        outputs = []
        for b in range(x['target_input_ids'].size(0)):
            output = self.generation_decoder(inputs_embeds=input_imbeds[b], attention_mask=x['target_attention_mask'][b].unsqueeze(0).expand(self.r_num,-1)) # (num_retrieval, answer_max_length, emb_dim=768)
            outputs.append(output.logits) # (num_retrieval, answer_max_length, vocab_size)
        outputs = torch.stack(outputs, dim=0) # (batch_size, num_retrieval, answer_max_length, vocab_size)
        
        # weighted mean outputs with score
        scores = self.softmax(torch.tensor(score)[:,:,None,None]) # (batch_size, num_retrieval, 1, 1)
        # (batch_size, num_retrieval, answer_max_length, vocab_size) -> (batch_size, answer_max_length, vocab_size)
        weighted_sum_outputs = torch.sum(scores*outputs.to('cpu'), dim=1) 

        return weighted_sum_outputs.to('cuda') # (batch_size, answer_max_length, vocab_size)

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch['target_input_ids'] # (batch_size, answer_max_length)

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
        y = batch['target_input_ids'] # (batch_size, answer_max_length)

        logits = self(**x)
        
        logits = logits.swapaxes(1, 2) # (batch_size, vocab_size, answer_max_length)
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss)

        # em metric
        predicts = torch.argmax(logits, dim=-2)
        for s, pred, y in zip(x['target_input_ids'], predicts,y):
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

    config = {"q_model_name":'klue/roberta-base',
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
    model = TrainModel(config["q_model_name"], config["gen_model_name"], config["learning_rate"], config["rm_num"])

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
