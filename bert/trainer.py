import os
import math
import time
import sys

import transformers
sys.path.append('bert')

import pandas as pd
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torchtext.legacy import data
from transformers import BertTokenizer, BertModel, AdamW, get_constant_schedule_with_warmup
from utils import save_logs, split_and_cut, convert_to_int
from model import BERTNLIModel
transformers.logging.set_verbosity_error()

class BERTNLITrainer:

    BATCH_SIZE = 16
    # HIDDEN_DIM = 512
    DROPOUT_RATE = 0.3
    OUTPUT_DIM = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.001
    EPSILON = 1e-6
    WARMUP_PERCENT = 0.2
    NUM_EPOCHS = 30
    SAVE_PATH_MODEL = './weight'
    PATIENCE = 10
    best_val_loss = None
    counter = 0

    def __init__(self):
        super(BERTNLITrainer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.label_data = data.LabelField()
        self.text_data = data.Field(batch_first = True,
                        use_vocab = False,
                        tokenize = split_and_cut,
                        preprocessing = self.tokenizer.convert_tokens_to_ids,
                        pad_token = self.tokenizer.pad_token_id,
                        unk_token = self.tokenizer.unk_token_id)
        self.attention_data = data.Field(batch_first = True,
                        use_vocab = False,
                        tokenize = split_and_cut,
                        preprocessing = convert_to_int,
                        pad_token = self.tokenizer.pad_token_id)
        self.token_type_data = data.Field(batch_first = True,
                        use_vocab = False,
                        tokenize = split_and_cut,
                        preprocessing = convert_to_int,
                        pad_token = 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_data_training(self):
        fields = [
            ('label', self.label_data), \
            ('sequence', self.text_data), \
            ('attention_mask', self.attention_data), \
            ('token_type', self.token_type_data) \
        ]
        self.train_data, self.valid_data = data.TabularDataset.splits(
            path='../input/contradictory-my-dear-watson-feature-engineering', 
            train='train_feature_engineering.csv',
            validation ='valid_feature_engineering.csv',
            # test='test_feature_engineering.csv',
            format='csv',
            fields=fields,
            skip_header=True
        )

        self.label_data.build_vocab(self.train_data)
        self.train_iterator = data.BucketIterator(
            (self.train_data), 
            batch_size = self.BATCH_SIZE,
            sort_key = lambda x: len(x.sequence),
            sort=False,
            shuffle=True,
            sort_within_batch = False, 
            device = self.device)
        
        self.valid_iterator = data.BucketIterator(
            (self.valid_data), 
            batch_size = self.BATCH_SIZE,
            sort_key = lambda x: len(x.sequence),
            sort=False,
            shuffle=False,
            sort_within_batch = False, 
            device = self.device)
        return self.train_iterator, self.valid_iterator

    def setup_data_inference(self):
        fields = [
            ('sequence', self.text_data), \
            ('attention_mask', self.attention_data), \
            ('token_type', self.token_type_data) \
            ]
        self.test_data = data.TabularDataset.splits(
            path='/kaggle/working',
            test='test_feature_engineering.csv',
            format='csv',
            fields=fields,
            skip_header=True
        )[0]

        self.test_iterator = data.BucketIterator(
            (self.test_data), 
            batch_size = self.BATCH_SIZE,
            sort_key = lambda x: len(x.sequence),
            sort=False,
            shuffle=False,
            sort_within_batch = False, 
            device = self.device)
        return self.test_iterator

    def __len__(self):
        print(f"Number of training data: {len(self.train_data)}")
        print(f"Number of validation data: {len(self.valid_data)}")
        # print(f"Number of testing data: {len(self.test_data)}")
        return len(self.train_data), len(self.valid_data)# , len(self.test_data)

    def compute_accuracy(self, preds, ground_truth):
        max_preds = preds.argmax(dim = 1, keepdim = True)
        correct = (max_preds.squeeze(1)==ground_truth).float()
        return correct.sum() / len(ground_truth)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def setup_training(self):
        self.model = BERTNLIModel(self.OUTPUT_DIM, self.DROPOUT_RATE)
        self.model.count_parameters()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE, eps=self.EPSILON)
        self.optimizer = AdamW(self.model.parameters(), lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY, eps=self.EPSILON, correct_bias=False)
        total_steps = math.ceil(self.NUM_EPOCHS*len(self.train_data)*1./self.BATCH_SIZE)
        warmup_steps = int(total_steps*self.WARMUP_PERCENT)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps)
        return self.model, self.criterion, self.optimizer, self.scheduler
    
    def early_stopping(self, val_loss, model):
        current_loss = val_loss
        early_stopping = False
        
        if self.best_val_loss is None:
            self.best_val_loss = current_loss
            if not os.path.exists(self.SAVE_PATH_MODEL):
                os.makedirs(self.SAVE_PATH_MODEL)
            torch.save(model.state_dict(), self.SAVE_PATH_MODEL + '/bert-nli.pt')
        elif current_loss > self.best_val_loss:
            self.counter += 1
            print(f'[Early Stopping Counter]: {self.counter} out of {self.PATIENCE}')
            if self.counter >= self.PATIENCE:
                early_stopping = True
        else:
            self.best_val_loss = current_loss
            if not os.path.exists(self.SAVE_PATH_MODEL):
                os.makedirs(self.SAVE_PATH_MODEL)
            torch.save(model.state_dict(), self.SAVE_PATH_MODEL + '/bert-nli.pt')
            self.counter = 0
        return early_stopping

    def train(self, model, iterator, criterion, optimizer, scheduler):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        
        for batch in iterator:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
            sequence = batch.sequence
            attn_mask = batch.attention_mask
            token_type = batch.token_type
            label = batch.label
            
            predictions = model(sequence, attn_mask, token_type)
            loss = criterion(predictions, label)
            accuracy = self.compute_accuracy(predictions, label)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
        return epoch_loss / len(iterator), epoch_accuracy / len(iterator)
    
    def evaluate(self, model, iterator, criterion):
        epoch_loss = 0
        epoch_accuracy = 0
        model.eval()
        
        with torch.no_grad():
            for batch in iterator:
                sequence = batch.sequence
                attn_mask = batch.attention_mask
                token_type = batch.token_type
                labels = batch.label
                            
                predictions = model(sequence, attn_mask, token_type)
                loss = criterion(predictions, labels)
                accuracy = self.compute_accuracy(predictions, labels)
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
        return epoch_loss / len(iterator), epoch_accuracy / len(iterator)
    
    def predict_submission(self, iterator):
        self.model = BERTNLIModel(self.OUTPUT_DIM, self.DROPOUT_RATE)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('../input/bert-nli/bert-nli.pt'))
        self.model.eval()
        predictions = []
        df_submission = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
        with torch.no_grad():
            for batch in iterator:
                sequence = batch.sequence
                attn_mask = batch.attention_mask
                token_type = batch.token_type
                            
                prediction = self.model(sequence, attn_mask, token_type)
                _, prediction = torch.max(prediction, dim=1)
                prediction = prediction.flatten().tolist()
                predictions += prediction
        
        df_submission['prediction'] = predictions
        df_submission.to_csv('submission.csv', index=False)
    
    def predict_submission_df(self):
        df_submission = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
        df_test = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BERTNLIModel(self.OUTPUT_DIM, self.DROPOUT_RATE)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('../input/bert-nli/bert-nli.pt'))
        self.model.eval()
        predictions = []
        premises = df_test['premise'].to_list()
        hypothesises = df_test['hypothesis'].to_list()
        
        for i in range(len(premises)):
            premise = '[CLS] ' + premises[i] + ' [SEP] '
            hypothesis = hypothesises[i] + ' [SEP]'

            premise_token = tokenizer.tokenize(premise)
            hypothesis_token = tokenizer.tokenize(hypothesis)

            premise_type = [0] * len(premise_token)
            hypothesis_type = [1] * len(hypothesis_token)

            indexes = premise_token + hypothesis_token
            indexes = tokenizer.convert_tokens_to_ids(indexes)
            indexes_type = premise_type + hypothesis_type
            attn_mask = [1] * len(indexes)
            indexes = torch.LongTensor(indexes).unsqueeze(0).to(self.device)
            indexes_type = torch.LongTensor(indexes_type).unsqueeze(0).to(self.device)
            attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(self.device)

            prediction = self.model(indexes, attn_mask, indexes_type)
            prediction = prediction.argmax(dim=-1).item()
            predictions.append(prediction)
        
        df_submission['prediction'] = predictions
        df_submission.to_csv('submission.csv', index=False)
    
    def predict_inference(self, premise, hypothesis):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        label = ['Entailment', 'Neutral', 'Contradiction']
        self.model = BERTNLIModel(self.OUTPUT_DIM, self.DROPOUT_RATE)
        # self.model.load_state_dict(torch.load('bert-nli.pt'))
        self.model.eval()
        
        premise = '[CLS] ' + premise + ' [SEP] '
        hypothesis = hypothesis + ' [SEP]'

        premise_token = tokenizer.tokenize(premise)
        hypothesis_token = tokenizer.tokenize(hypothesis)

        premise_type = [0] * len(premise_token)
        hypothesis_type = [1] * len(hypothesis_token)

        indexes = premise_token + hypothesis_token
        indexes = tokenizer.convert_tokens_to_ids(indexes)
        attn_mask = [1] * len(indexes)
        indexes_type = premise_type + hypothesis_type
        
        indexes = torch.LongTensor(indexes).unsqueeze(0).to(self.device)
        attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(self.device)
        indexes_type = torch.LongTensor(indexes_type).unsqueeze(0).to(self.device)
        
        prediction = self.model(indexes, attn_mask, indexes_type)
        prob, pred = torch.max(prediction.softmax(dim=1), dim=1)
        return (premise, hypothesis), (indexes, attn_mask, indexes_type), (prob.item(), label[pred])

    def train_process(self, model, train_iterator, valid_iterator, optimizer, criterion, scheduler):
        start_time = time.time()
        early_stopping = False
        
        # For progress record.
        train_loss_logs = np.zeros(shape=self.NUM_EPOCHS, dtype=np.float)
        eval_loss_logs = np.zeros(shape=self.NUM_EPOCHS, dtype=np.float)
        train_accuracy_logs = np.zeros(shape=self.NUM_EPOCHS, dtype=np.float)
        eval_accuracy_logs = np.zeros(shape=self.NUM_EPOCHS, dtype=np.float)

        for epoch in tqdm(range(self.NUM_EPOCHS)):
            train_loss, train_acc = self.train(model, train_iterator, criterion, optimizer, scheduler)
            valid_loss, valid_acc = self.evaluate(model, valid_iterator, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')

            train_loss_logs[epoch] = train_loss
            train_accuracy_logs[epoch] = train_acc
            eval_loss_logs[epoch] = valid_loss
            eval_accuracy_logs[epoch] = valid_acc
            
            early_stopping = self.early_stopping(valid_loss, model)
            if early_stopping:
                print("Early stopping")
                break
            
        logs = [train_loss_logs, eval_loss_logs, train_accuracy_logs, eval_accuracy_logs]
        save_logs(logs=logs, NUM_EPOCHS=self.NUM_EPOCHS)
