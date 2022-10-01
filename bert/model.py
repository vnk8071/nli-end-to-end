import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTNLIModel(nn.Module):
    def __init__(self, output_dim, dropout_rate):
        super(BERTNLIModel, self).__init__()
        self.bertnli_model = SingleBERT(use_pooling=False)
        self.bert_config = self.bertnli_model.get_config()
        self.hidden_size = self.bert_config.hidden_size
        self.output_dim = output_dim
        self.classifier = Classifier(hidden_size=self.hidden_size, num_classes=self.output_dim, dropout_rate=dropout_rate)

    def forward(self, sequence, attention_mask, token_type):
        output = self.bertnli_model(input=sequence, attention_mask=attention_mask, token_type_ids=token_type)
        output = self.classifier(output)
        return output

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.bertnli_model.parameters() if p.requires_grad)
        return print(f'The model has {total_params:,} trainable parameters')

class SingleBERT(nn.Module):
    def __init__(self, use_pooling=False):
        super(SingleBERT, self).__init__()
        self.use_pooling = use_pooling
        self.bert_config = BertConfig.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

        for param in self.bert_model.parameters():
            param.requires_grad = True

    def forward(self, input, attention_mask, token_type_ids):
        last_layer_output, pooling_output = self.bert_model(input, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        if self.use_pooling:
            return pooling_output
        return last_layer_output[:, 0, :]

    def get_config(self):
        return self.bert_config

        
class Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_rate):
        super(Classifier, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.batchNorm = nn.BatchNorm1d(num_features=hidden_size, eps=1e-05, momentum=0.1, affine=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

        nn.init.normal_(self.linear1.weight, std=0.04)
        nn.init.normal_(self.linear2.weight, mean=0.5, std=0.04)
        nn.init.normal_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.bias, 0)

    def forward(self, input):
        output = self.dropout1(input)
        output = self.linear1(output)
        output = self.batchNorm(output)
        output = self.activation(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        return output
