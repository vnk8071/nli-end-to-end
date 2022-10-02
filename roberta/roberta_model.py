import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, XLMRobertaModel, XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AdamW

class XLMRoberta_Arch(nn.Module):
    
    def __init__(self, model_name, n_classes, freeze_bert=None):
        
        super(XLMRoberta_Arch, self).__init__()
        self.roberta =  XLMRobertaModel.from_pretrained(
                                                        model_name,
                                                        num_labels = n_classes, 
                                                        return_dict=False
                                                        )
        if freeze_bert:
            for p in self.roberta.parameters():
                p.requires_grad = False
                
        self.bert_drop_1 = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size) # (768, 64)
        self.bn = nn.BatchNorm1d(768) # (768)
        self.bert_drop_2 = nn.Dropout(0.25)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes) # (768,3)
    
    def forward(self, input_ids, attention_mask):
        _, output = self.roberta(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        output = self.bert_drop_1(output)
        output = self.fc(output)
        output = self.bn(output)
        output = self.bert_drop_2(output)
        output = self.out(output)        
        return output
    
if __name__ == '__main__':
    
    import pandas as pd
    from transformers import AutoTokenizer, XLMRobertaModel, XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AdamW
    from roberta.roberta_dataset import SherlockDataset
    
    MAX_LENGTH = 250
    MODEL_NAME='xlm-roberta-base'
    
    model = XLMRoberta_Arch(MODEL_NAME, 3)
    TOKENIZER = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    
    train_df = pd.read_csv('./data/train.csv')
    train_data = SherlockDataset(train_df, tokenizer=TOKENIZER)
    train_loader = train_data.create_dataloader(batch_size=8)
    input_ids, att_mask, labels = next(iter(train_loader))
    
    output = model(input_ids, att_mask)
    model.eval()
    a = model(input_ids[0].unsqueeze(0), att_mask[0].unsqueeze(0))
    print()