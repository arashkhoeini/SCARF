from torch.utils.data import DataLoader, Dataset, Subset
import torch
from model.epoch_sampler import EpochSampler
import pandas as pd

def init_data_loaders(dataset: Dataset, batch_size, train_split=0.7, val_split=0.1, labeled_ratio=1, class_aware = False):

    if not class_aware:
        pretrain_idx = torch.arange(0,int(len(dataset)*train_split))
        finetuning_idx = torch.arange(0,int(len(pretrain_idx)*labeled_ratio))
        val_idx = torch.arange(int(len(dataset)*train_split), int(len(dataset)*(train_split+val_split)))
        test_idx = torch.arange(int(len(dataset)*(train_split+val_split)), len(dataset))
    else:
        target = torch.tensor([dataset.class_transform_dict[label] for label in dataset.y])
        uniq = torch.unique(target, sorted=True)
        class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
        class_idxs = [idx[torch.randperm(len(idx))] for idx in class_idxs]
        pretrain_idx = torch.cat([idx[:int(train_split*len(idx))] for idx in class_idxs])
        finetuning_idx = torch.cat([idx[:int(labeled_ratio*len(idx))] for idx in class_idxs])
        val_idx = torch.cat([idx[int(train_split*len(idx)):int((train_split+val_split)*len(idx))] for idx in class_idxs])
        test_idx = torch.cat([idx[int((train_split+val_split)*len(idx)):] for idx in class_idxs])
    
    # print('dataset size: ', len(dataset))
    # print('pretraining size: ', len(pretrain_idx))
    # print('finetuning size: ', len(finetuning_idx))
    # print('validation size: ', len(val_idx))
    # print('test size: ', len(test_idx))
    
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