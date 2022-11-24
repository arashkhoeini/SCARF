import torch
import numpy as np
from configs.init_configs import init_config
import sys
import os
from pathlib import Path
import pandas as pd
from data.dataset import TabularDataset
from data.utils import preprocess
from trainer import Trainer

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_datasets(path):
    path = Path(path)
    datasets = []
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            df = pd.read_csv(path/file_name)
            df = preprocess(df)
            # y = df.pop('class')
            # dataset = TabularDataset(df.values, y.values, df.shape[1], len(y.unique()))
            datasets.append(df)
        break
    return datasets
    

def main():
    configs = init_config("configs/openml.yml", sys.argv)
    if configs.fix_seed:
        init_seed(configs.seed)
    datasets = prepare_datasets(configs.data_dir)
    for dataset in datasets:
        trainer = Trainer(dataset, configs)
        trainer.run()
    

if __name__ == '__main__':
    main()