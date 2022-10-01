import re

import pandas as pd
import numpy as np

from torchtext.legacy import data
from transformers import BertTokenizer
from deep_translator import GoogleTranslator
from dask import bag, diagnostics

class FeatureEngineeringNLI:

    TRIM_CHARACTER = 128

    def __init__(self, df_path: str, is_train: bool):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.is_train = is_train
        self.tokenizer = BertTokenizer.from_pretrained('../input/bert-base-multilingual-cased/bert-base-multilingual-cased', local_files_only=True)

    def tokenize_bert(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        return tokens
    
    def trim_sentence(self, sentence):
        try:
            sent = re.sub(r'\(|\)|\[|\]|\{|\}', '', sentence)
            sent = sent.split()
            sent = sent[:self.TRIM_CHARACTER]
            return " ".join(sent)
        except:
            return sentence
    
    def premise_token_type(self, sentence):
        return [0] * len(sentence)
    
    def hypothesis_token_type(self, sentence):
        return [1] * len(sentence)

    def combine_sequence(self, sentence):
        return " ".join(sentence)

    def combine_mask(self, mask):
        mask = [str(m) for m in mask]
        return " ".join(mask)
    
    #TODO: use a dask dataframe instead of all this
    def trans_parallel(self, df, translator):
        premise_bag = bag.from_sequence(df.premise.tolist()).map(translator.translate)
        hypo_bag =  bag.from_sequence(df.hypothesis.tolist()).map(translator.translate)
        with diagnostics.ProgressBar():
            premises = premise_bag.compute()
            hypos = hypo_bag.compute()
        df[['premise', 'hypothesis']] = list(zip(premises, hypos))
        return df
    
    def english_translate(self, df):
        translator = GoogleTranslator(target='en')
        df[df.lang_abv != "en"] =  df.loc[df.lang_abv != "en"].copy().pipe(self.trans_parallel, translator)
        df['lang_abv'] = ['en']*len(df)
        df['language'] = ['English']*len(df)
        return df
    
    def process_data(self):
        self.df = self.english_translate(self.df)
        self.df['premise_trim'] = self.df['premise'].apply(self.trim_sentence)
        self.df['hypothesis_trim'] = self.df['hypothesis'].apply(self.trim_sentence)

        self.df['premise_format'] = '[CLS] '  + self.df['premise_trim'] + ' [SEP] '
        self.df['hypothesis_format'] = self.df['hypothesis_trim'] + ' [SEP]'

        self.df['premise_tokenizer'] = self.df['premise_format'].apply(self.tokenize_bert)
        self.df['hypothesis_tokenizer'] = self.df['hypothesis_format'].apply(self.tokenize_bert)
        self.df['sequence'] = self.df['premise_tokenizer'] + self.df['hypothesis_tokenizer']

        self.df['premise_token_type'] = self.df['premise_tokenizer'].apply(self.premise_token_type)
        self.df['hypothesis_token_type'] = self.df['hypothesis_tokenizer'].apply(self.hypothesis_token_type)
        self.df['token_type'] = self.df['premise_token_type'] + self.df['hypothesis_token_type']

        self.df['attention_mask'] = self.df['sequence'].apply(self.hypothesis_token_type)
        self.df['attention_mask'] = self.df['attention_mask'].apply(self.combine_mask)
        self.df['token_type'] = self.df['token_type'].apply(self.combine_mask)
        self.df['sequence'] = self.df['sequence'].apply(self.combine_sequence)
        
        if self.is_train:
            self.df = self.df[['label', 'sequence', 'attention_mask', 'token_type']]
            random_state = np.random.RandomState()
            train = self.df.sample(frac=0.8, random_state=random_state)
            train.to_csv('train_feature_engineering.csv', index=False)
            valid = self.df.loc[~self.df.index.isin(train.index)]
            valid.to_csv('valid_feature_engineering.csv', index=False)
        else:
            self.df = self.df[['sequence', 'attention_mask', 'token_type']]
            self.df.to_csv('test_feature_engineering.csv', index=False)