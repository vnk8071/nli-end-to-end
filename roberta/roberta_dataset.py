import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SherlockDataset(Dataset):
    def __init__(self, df, tokenizer=None, max_length=256, test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test = test
        
    def __getitem__(self, index):
        
        sentence1 = self.df.loc[index, 'premise']
        sentence2 = self.df.loc[index, 'hypothesis']
        
        encoded_dict = self.tokenizer.encode_plus(
                sentence1, sentence2,
                padding='max_length',
                add_special_tokens = True,
                max_length = self.max_length,     
                truncation = True,
                return_attention_mask = True,   
                return_tensors = 'pt' # return pytorch tensors
       )
        
        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        if not self.test:
            target = torch.tensor(self.df.loc[index, 'label'])
            return (input_ids, attention_mask, target)
        else:
            return (input_ids, attention_mask)
    
    def __len__(self):
        return len(self.df)


    def create_dataloader(self, batch_size, shuffle=False, num_workers=0, drop_last=False):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=False)
        
if __name__ == '__main__':
    
    import pandas as pd
    from transformers import AutoTokenizer, XLMRobertaModel, XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AdamW    
    MAX_LENGTH = 250
    MODEL_NAME='xlm-roberta-base'

    TOKENIZER = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    train_df = pd.read_csv('./data/train.csv')
    train_data = SherlockDataset(train_df, tokenizer=TOKENIZER, max_length=MAX_LENGTH)
    train_loader = train_data.create_dataloader(batch_size=32)
    
    print()
    