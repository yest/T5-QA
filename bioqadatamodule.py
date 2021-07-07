import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl


class BioQADataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 396,
        target_max_token_len: int = 32
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        
        source_encoding = self.tokenizer(
            data_row['question'],
            data_row['context'],
            max_length = self.source_max_token_len,
            padding = 'max_length',
            truncation = 'only_second',
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
        )
        try:
            target_encoding = self.tokenizer(
                data_row['answer_text'],
                max_length = self.target_max_token_len,
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,
                add_special_tokens = True,
                return_tensors = 'pt'
            )
        except:
            print(data_row['answer_text'])
        
        labels = target_encoding['input_ids']
        labels[labels == 0] = -100
        
        return dict(
            question = data_row['question'],
            context = data_row['context'],
            answer_text = data_row['answer_text'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels = labels.flatten()
        )
    

class BioQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        source_max_token_len: int = 396,
        target_max_token_len: int = 32        
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        
        self.train_dataset = BioQADataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.val_dataset = BioQADataset(
            self.val_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )
        
        self.test_dataset = BioQADataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = 1,
            num_workers = 4,
            drop_last=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = 1,
            num_workers = 4,
            drop_last=True
        )