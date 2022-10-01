import argparse
import time

from trainer import BERTNLITrainer
from deep_translator import GoogleTranslator

parser = argparse.ArgumentParser()
parser.add_argument('--premise', type=str, default='My own little corner of the world, policy working, is an example.')
parser.add_argument('--hypothesis', type=str, default='An example is policy working.,en,English')
args = parser.parse_args()

def bert_inference(premise=args.premise, hypothesis=args.hypothesis):
    start = time.time()
    # translator = GoogleTranslator(target='en')
    print(f"Original premise: {premise}")
    print(f"Original hypothesis: {hypothesis}")
    print('-'*50)

    # premise_translated = translator.translate(args.premise)
    # hypothesis_translated = translator.translate(args.hypothesis)
    # print(f"Translated premise: {premise_translated}")
    # print(f"Translated hypothesis: {hypothesis_translated}")
    print('-'*50)

    predictor = BERTNLITrainer()
    (premise_, hypothesis_), (index, mask, token_type), (prob, pred) = predictor.predict_inference(args.premise, args.hypothesis)
    end = time.time() - start
    print(f"Total time predict: {end} second")
    return (premise_, hypothesis_), (index, mask, token_type), (prob, pred)

if __name__ == '__main__':
    (premise, hypothesis), (index, mask, token_type), (prob, pred) = bert_inference(args.premise, args.hypothesis)
    print(f"Probability: {prob:.2f} - Prediction: {pred}")