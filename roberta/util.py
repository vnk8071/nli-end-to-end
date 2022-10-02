from nltk.corpus import stopwords
import re

STOPWORDS = stopwords.words("english")

def clean_text(text, stopwords=STOPWORDS):
    text = text.lower()
    
    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)
    
    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()
    
    return text

def remove_punctuation(sentence):
    sentence = re.sub(r"[()\"#/@;:<>{}`+=~|.!?,]", " ", sentence)
    return sentence


def clean_sentence(sentence, stopwords=STOPWORDS):
    sentence = sentence.lower()
    
    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    sentence = pattern.sub("", sentence)
    
    # Spacing and filters
    sentence = re.sub(" +", " ", sentence)  # remove multiple spaces
    sentence = sentence.strip()
    return sentence