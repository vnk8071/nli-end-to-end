import os
import re
import math
import time
import datetime

from trainer import BERTNLITrainer
from feature_engineering import FeatureEngineeringNLI

if __name__ == '__main__':
    df_train_path = ""
    print("Start feature engineering process")
    feature_engineering_train = FeatureEngineeringNLI(df_train_path, is_train=True)
    feature_engineering_train.process_data()

    print("Setup and training")
    trainer = BERTNLITrainer()
    train_iterator, valid_iterator = trainer.setup_data_training()
    model, criterion, optimizer, scheduler = trainer.setup_training()
    trainer.train_process(model, train_iterator, valid_iterator, optimizer, criterion, scheduler)

    