import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from deep_translator import GoogleTranslator
from transformers import XLMRobertaModel, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from roberta_model import XLMRoberta_Arch
from util import remove_punctuation, clean_sentence
from nltk.corpus import stopwords
import argparse

MODEL_NAME='xlm-roberta-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH_MODEL = './roberta/weight'
TOKENIZER = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
MAX_LENGTH = 256
STOPWORDS = stopwords.words("english")

parser = argparse.ArgumentParser()
parser.add_argument('--premise', type=str, default='My own little corner of the world, policy working, is an example.')
parser.add_argument('--hypothesis', type=str, default='An example is policy working.,en,English')
args = parser.parse_args()

def roberta_inference(premise=args.premise, hypothesis=args.hypothesis):
    class_to_index = ['Entailment', 'Neutral','Contracdiction']
    model = XLMRoberta_Arch(model_name=MODEL_NAME, n_classes=3).to(device)
    model.load_state_dict(torch.load(SAVE_PATH_MODEL + "/model.pt", map_location=torch.device('cpu')))
    model.to(device)
    
    translator = GoogleTranslator(target='en')
    print(f"Original premise: {premise}")
    print(f"Original hypothesis: {hypothesis}")
    print('-'*50)
    
    premise_rp = remove_punctuation(premise)
    hypothesis_rp = remove_punctuation(hypothesis)
    
    premise_translated = translator.translate(premise_rp)
    hypothesis_translated = translator.translate(hypothesis_rp)
    
    premise_clean = clean_sentence(premise_translated, STOPWORDS)
    hypothesis_clean = clean_sentence(hypothesis_translated, STOPWORDS)
    
    sentence1 = premise_clean
    sentence2 = hypothesis_clean
            
    encoded_dict = TOKENIZER.encode_plus(
            sentence1, sentence2,
            padding='max_length',
            add_special_tokens = True,
            max_length = MAX_LENGTH,     
            truncation = True,
            return_attention_mask = True,   
            return_tensors = 'pt' # return pytorch tensors
    )

    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    
    model.eval()
    output = model(input_ids.to(device), attention_mask.to(device),
                                )
    pred = torch.argmax(output, 1).item()
    probs = (F.softmax(output)).max().item()
    return (sentence1, sentence2), (input_ids, attention_mask), (class_to_index[pred], probs)

if __name__ == '__main__':
    premise = 'Then I considered'
    hypothesis = 'I refused to even consider it'
    (sentence1, sentence2), (input_ids, attention_mask), (pred, probs) = roberta_inference(args.premise, args.hypothesis)
    
    print()