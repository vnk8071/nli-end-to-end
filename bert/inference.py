import argparse
import time

from trainer import BERTNLITrainer
from deep_translator import GoogleTranslator
import nltk
nltk.download('stopwords')
from utils import remove_punctuation, clean_sentence
from nltk.corpus import stopwords

STOPWORDS = stopwords.words("english")

parser = argparse.ArgumentParser()
parser.add_argument('--premise', type=str, default='My own little corner of the world, policy working, is an example.')
parser.add_argument('--hypothesis', type=str, default='An example is policy working.,en,English')
args = parser.parse_args()

def bert_inference(premise=args.premise, hypothesis=args.hypothesis):
    start = time.time()
    translator = GoogleTranslator(target='en')
    print(f"Original premise: {premise}")
    print(f"Original hypothesis: {hypothesis}")
    print('-'*50)

    premise_rp = remove_punctuation(premise)
    hypothesis_rp = remove_punctuation(hypothesis)
    print(f"Premise remove punctuation: {premise_rp}")
    print(f"Hypothesis remove punctuation: {hypothesis_rp}")
    print('-'*50)

    premise_translated = translator.translate(premise_rp)
    hypothesis_translated = translator.translate(hypothesis_rp)
    print(f"Translated premise: {premise_translated}")
    print(f"Translated hypothesis: {hypothesis_translated}")
    print('-'*50)

    premise_clean = clean_sentence(remove_punctuation(premise_translated), STOPWORDS)
    hypothesis_clean = clean_sentence(remove_punctuation(hypothesis_translated), STOPWORDS)
    print(f"Premise after clean: {premise_clean}")
    print(f"Hypothesis after clean: {hypothesis_clean}")
    print('-'*50)

    predictor = BERTNLITrainer()
    (premise_, hypothesis_), (index, mask, token_type), (prob, pred) = predictor.predict_inference(premise_clean, hypothesis_clean)
    end = time.time() - start
    print(f"Total time predict: {end} second")
    return (premise_rp, hypothesis_rp), (premise_translated, hypothesis_translated), (premise_clean, hypothesis_clean), (premise_, hypothesis_), (index, mask, token_type), (prob, pred)

if __name__ == '__main__':
    (premise_rp, hypothesis_rp), (premise_translated, hypothesis_translated), (premise_clean, hypothesis_clean), (premise, hypothesis), (index, mask, token_type), (prob, pred) = bert_inference(args.premise, args.hypothesis)
    print(f"Probability: {prob:.2f} - Prediction: {pred}")