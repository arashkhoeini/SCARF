from torch.utils.data import DataLoader, Dataset, Subset
import torch
from model.epoch_sampler import EpochSampler
import pandas as pd

def init_data_loaders(dataset: Dataset, batch_size, train_split=0.7, val_split=0.1, labeled_ratio=1):

    pretrain_idx = torch.arange(0,int(len(dataset)*train_split))
    finetuning_idx = torch.arange(0,int(len(pretrain_idx )*labeled_ratio))
    val_idx = torch.arange(int(len(dataset)*train_split), int(len(dataset)*(train_split+val_split)))
    test_idx = torch.arange(int(len(dataset)*(train_split+val_split)), len(dataset))

    pretrain_loader = DataLoader(Subset(dataset, pretrain_idx), batch_size=batch_size)
    finetuning_loader = DataLoader(Subset(dataset, finetuning_idx), batch_size=batch_size)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

    return pretrain_loader, finetuning_loader, val_loader, test_loader

def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    # TODO: Complete
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index()
    return  dataframe