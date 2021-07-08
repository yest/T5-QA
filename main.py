from argparse import ArgumentParser

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from bioqamodel import BioQAModel
from bioqadatamodule import BioQADataModule

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


def main(hparams):
    pl.seed_everything(22)

    DATASET_PATH = 'data/bioasq/'
    MODEL_NAME = hparams.model
    TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)
    BATCH_SIZE = hparams.batch_size
    EPOCHS = hparams.epochs

    train_df = pd.read_csv(f"{DATASET_PATH}train.tsv", sep='\t').dropna()
    val_df = pd.read_csv(f"{DATASET_PATH}val.tsv", sep='\t')
    test_df = pd.read_csv(f"{DATASET_PATH}test.tsv", sep='\t')


    data_module = BioQADataModule(train_df, val_df, test_df, TOKENIZER, batch_size=BATCH_SIZE)
    model = BioQAModel(MODEL_NAME)

    checkpoint_callback = ModelCheckpoint(
        dirpath = "checkpoints",
        filename = "best-checkpoint",
        save_top_k = 1,
        verbose = True,
        monitor = "val_loss",
        mode = "min"
    )   

    trainer = Trainer(
    #     logger=logger
        callbacks=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30 
    )

    trainer.fit(model, data_module)
    trainer.test()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='t5-base')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=6)
    args = parser.parse_args()

    main(args)