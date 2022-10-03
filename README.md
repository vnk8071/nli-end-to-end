# Natural Language Inference with Transformers, Flask, Docker, AWS

## Overview

The Challenge:
If you have two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. <b>Natural Language Inference (NLI)</b> is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related.

The task is to create an NLI model that assigns labels of <b>0, 1, or 2 (corresponding to entailment, neutral, and contradiction)</b> to pairs of premises and hypotheses. To make things more interesting, the train and test set include text in <b>fifteen different languages!</b>

For more details: https://www.kaggle.com/competitions/contradictory-my-dear-watson/overview

## Install packages
Create virtual environment with conda
```
conda create -n nli python=3.8
conda activate nli
```
And then
```
pip install -r requirements.txt
```

## Download pre-trained models
```
bash download_pretrained_model.sh
```

## BERT
### Feature engineering:
```
python bert/feature_engineering.py
```
### Train:
```
python bert/train.py
```
### Inference:
```
python bert/inference.py --premise "My own little corner of the world, policy working, is an example." --hypothesis "An example is policy working.,en,English"
```
or simple with
```
python bert/inference.py
```

## RoBERTa
```
python roberta/roberta_inference.py --premise "<premise>" --hypothesis "<hypotheis>"
python roberta/roberta_inference.py --premise "My own little corner of the world, policy working, is an example." --hypothesis "An example is policy working.,en,English"
```
or simple with
```
python roberta/roberta_inference.py
```
## API
Run app Flask
```
python app/views.py
```
Open UI of app: http://127.0.0.1:5000/

## Docker (Coming soon)

## Amazon Web Service (AWS) Coming soon

