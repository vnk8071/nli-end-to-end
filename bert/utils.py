import os
import datetime
import re

import matplotlib.pyplot as plt

MAX_INPUT_LENGTH = 512

def convert_to_int(tok_ids):
    tok_ids = [int(x) for x in tok_ids]
    return tok_ids

def split_and_cut(sentence):
    tokens = sentence.strip().split(" ")
    tokens = tokens[:MAX_INPUT_LENGTH]
    return tokens

def trim_sentence(sentence):
    try:
        sent = re.sub(r'\(|\)|\[|\]|\{|\}|\"', '', sentence)
        return " ".join(sent)
    except:
        return sentence

def remove_punctuation(sentence):
    sentence = re.sub(r"[()\"#/@;:<>{}`+=~|.!?,]", " ", sentence)
    return sentence


def clean_sentence(sentence, stopwords):
    sentence = sentence.lower()
    
    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    sentence = pattern.sub("", sentence)
    
    # Spacing and filters
    sentence = re.sub(" +", " ", sentence)  # remove multiple spaces
    sentence = sentence.strip()
    return sentence

def save_logs(logs, NUM_EPOCHS):
    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'

    save_logs_path = './logs/' + str(time_info)
    if not os.path.exists(save_logs_path):
        os.makedirs(save_logs_path)

    save_path_loss = save_logs_path  + '/loss' + '.jpg'
    save_path_acurracy = save_logs_path  + '/accuracy' + '.jpg'

    x = [num for num in range(NUM_EPOCHS)]
    epoch_train_losses = logs[0].tolist()
    epoch_eval_losses = logs[1].tolist()
    epoch_train_accuracies = logs[2].tolist()
    epoch_eval_accuracies = logs[3].tolist()

    # Plot Loss
    plt.plot(x, epoch_train_losses, color='red', label='Train Loss')
    plt.plot(x, epoch_eval_losses, color='blue', label='Eval Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig(save_path_loss)
    plt.clf()

    # Plot Accuracy
    plt.plot(x, epoch_train_accuracies, color='red', label='Train Accuracy')
    plt.plot(x, epoch_eval_accuracies, color='blue', label='Eval Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
 
    plt.savefig(save_path_acurracy)
    
    return None