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
    dataset = pd.read_csv(path, index_col='Unnamed: 0')
    return dataset
    

def main():
    configs = init_config("configs/sc.yml", sys.argv)
    
    if configs.fix_seed:
        init_seed(configs.seed)
    dataset = prepare_datasets(configs.data_dir)
    
    trainer = Trainer(dataset, configs)
    trainer.run()
    

if __name__ == '__main__':
    main()