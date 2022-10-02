from roberta_trainer import Trainer
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch.optim as optim

from roberta_dataset import SherlockDataset
from roberta_model import XLMRoberta_Arch

NUM_CLASSES = 3
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_WORKERS = 1
LEARNING_RATE = 2e-5
PATIENCE = 3
EPOCHS = 5
MODEL_NAME='xlm-roberta-base'
TOKENIZER = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
SAVE_PATH_MODEL = './weight'

def train():
    model = XLMRoberta_Arch(MODEL_NAME, n_classes=NUM_CLASSES)
    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
    trainer = Trainer(
    model=model, device=device, optimizer=optimizer,
    loss_fn=loss_criteria, scheduler=None)

    train_data_loader, val_data_loader = trainer.set_up_training_data()
    best_model = trainer.train(EPOCHS, PATIENCE, train_data_loader, val_data_loader)
    
    return best_model

if __name__ == '__main__':
    best_model = train()