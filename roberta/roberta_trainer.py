import sys
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append('roberta')
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaModel, XLMRobertaForSequenceClassification, XLMRobertaTokenizer

from util import clean_text
from roberta_dataset import SherlockDataset
from roberta_model import XLMRoberta_Arch

NUM_CLASSES = 3
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_WORKERS = 1
LEARNING_RATE = 2e-5
PATIENCE = 3
EPOCHS = 5
MODEL_NAME='xlm-roberta-base'
TOKENIZER = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
SAVE_PATH_MODEL = './weight'

class Trainer(object):
    
    def __init__(self, model, device, optimizer, loss_fn, scheduler=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train_epoch(self, data_loader):
        self.model.train()
        train_loss = 0
        torch.set_grad_enabled(True)
        pbar = tqdm(enumerate(data_loader), total = len(data_loader))
        for i, batch in pbar:

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            self.model.zero_grad()
            output = self.model(b_input_ids, b_input_mask,
                            )
#             output = output.logits
            loss = self.loss_fn(output, b_labels)
            
            self.optimizer.zero_grad()

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            train_loss += loss.item()
#             print('loss batch:', loss)
#         del b_input_ids, b_input_mask, b_labels, loss
#         print(f'Average Training loss: {round(train_loss / len(data_loader), 5)}')
            train_desc = f'Loss: {train_loss/(i+1):.4f}'
            pbar.set_description(desc = train_desc)
        return train_loss / len(data_loader)
    
    def eval_epoch(self, data_loader):
        self.model.eval()
        torch.set_grad_enabled(False)
        eval_loss = 0
        targets_list = []
        with torch.inference_mode():
            pbar = tqdm(enumerate(data_loader), total = len(data_loader))
            for i, batch in pbar:

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                output = self.model(b_input_ids, b_input_mask,
                             )
#                 output = output.logits
                loss = self.loss_fn(output, b_labels)
                eval_loss += loss.item()

                preds = output

                # Move preds to the CPU
                val_preds = preds.detach().cpu().numpy()
                
                # Move the labels to the cpu
                targets_np = b_labels.to('cpu').numpy()

                targets_list.extend(targets_np)

                if i == 0:  # first batch
                    stacked_val_preds = val_preds

                else:
                    stacked_val_preds = np.vstack((stacked_val_preds, val_preds))
                val_desc = f'Loss: {eval_loss/(i+1):.4f}'
                pbar.set_description(desc = val_desc)
        # Calculate the validation accuracy
        y_true = targets_list
        y_pred = np.argmax(stacked_val_preds, axis=1)
        print(y_pred)
        val_acc = accuracy_score(y_true, y_pred)
        
        
#         print('Average Val loss:' , eval_loss/len(data_loader))
#         print('Val acc: ', val_acc)

#         del b_input_ids, b_input_mask, b_labels, loss
		
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return eval_loss/len(data_loader), val_acc
    
    def train(self, epochs, patience, train_loader, val_loader):
        best_val_loss = np.inf
        for epoch in range(epochs):

            avg_train_loss = self.train_epoch(data_loader=train_loader)
            avg_val_loss, val_acc = self.eval_epoch(data_loader=val_loader)
            if self.scheduler:    
                self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = self.model
                torch.save(self.model.state_dict(), self.SAVE_PATH_MODEL + '/roberta_nli.pt')
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                print("Stopping early!")
                break

            print(
                f"Epoch: {epoch} | "
                f"train_loss: {avg_train_loss:.5f}, "
                f"val_loss: {avg_val_loss:.5f}, "
                f"val_acc: {val_acc:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )

        return best_model

    def set_up_training_data(self):
        
        print('----- Setting up data ... -----')
        train_df = pd.read_csv('./data/train.csv')
        train_df[train_df['lang_abv']=='env']['premise'] = train_df[train_df['lang_abv']=='env']['premise'].apply(clean_text)
        train_df[train_df['lang_abv']=='env']['hypothesis'] = train_df[train_df['lang_abv']=='env']['hypothesis'].apply(clean_text)
        
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        train_df.reset_index(inplace=True)
        val_df.reset_index(inplace=True)
        
        train_data = SherlockDataset(train_df, tokenizer=TOKENIZER)
        val_data = SherlockDataset(val_df, tokenizer=TOKENIZER)
        
        train_data_loader = train_data.create_dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        val_data_loader = val_data.create_dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        print('Done')
        
        return train_data_loader, val_data_loader
    
        
    