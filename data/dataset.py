from cmath import exp
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class TabularDataset(torch.utils.data.Dataset):

    """
    Dataset for reading experiment matrices ([cell,gene] matrix)
    Parameters
    __________
    x: Tensor
    y: Tensor
    """
    def __init__(self, df):
        super().__init__()
        self.dataset_size= df.shape[0]
        self.x_cat, self.x_num, self.one_hot_encoder, self.input_dim, self.y = self.prepare_dataset(df)
        self.n_classes = len(self.y.unique())
        self.class_transform_dict = {c:i for i,c in enumerate(self.y.unique())}
        # print("Dataset Stats:")
        # print("\t")

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            item = item.item()
        
        x_cat_corrupted, x_num_corrupted = self.corrupt(self.x_cat[item], self.x_num[item])
        x = np.concatenate((self.one_hot_encoder.transform([self.x_cat[item]])[0], self.x_num[item]))
        x_corrupted = np.concatenate((self.one_hot_encoder.transform([x_cat_corrupted])[0], x_num_corrupted))
        
        return x, x_corrupted, self.class_transform_dict[self.y[item]]

    def __len__(self):
        return self.dataset_size

    def prepare_dataset(self, df):
        y = df.pop('class')
        df_categorical = df.select_dtypes(include='object')
        df_numerical = df.select_dtypes(exclude='object')
        one_hot_encoder = OneHotEncoder(sparse=False).fit(df_categorical.values)
        input_dim = sum([len(l) for l in one_hot_encoder.categories_])+df_numerical.shape[1]
        return df_categorical.values, df_numerical.values, one_hot_encoder, input_dim, y

    def corrupt(self, x_categorical, x_numerical, corruption_rate=0.6):
        x_cat_corrputed = x_categorical.copy()
        x_num_corrupted = x_numerical.copy()
        
        mask = np.random.rand(len(x_categorical)) < corruption_rate
        random_idx = np.random.randint(self.x_cat.shape[0], size=sum(mask))
        x_cat_corrputed[mask] = self.x_cat[random_idx, mask]

        mask = np.random.rand(len(x_numerical)) < corruption_rate
        random_idx = np.random.randint(self.x_num.shape[0], size=sum(mask))
        x_num_corrupted[mask] = self.x_num[random_idx, mask]
        
        return x_cat_corrputed, x_num_corrupted


class SingleCellDataset(torch.utils.data.Dataset):
    """
    Dataset for reading single cell experiments ([cell,gene] matrix)

    Parameters
    __________
    x: Tensor

    cells: ndarray

    genes: ndarray

    celltype: ndarray
    """
    def __init__(self, dataset_df):
        super().__init__()
        
        self.y, self.cells, self.genes, self.x, self.input_dim = self.prepare_datasaet(dataset_df)
        self.n_classes = len(set(self.y))
        self.class_transform_dict = {c:i for i,c in enumerate(set(self.y))}
    
    def prepare_datasaet(self, dataset_df):

        y = dataset_df.pop('class').values
        cells = dataset_df.index.values
        genes = dataset_df.columns
        x = dataset_df.values
        input_dim = x.shape[1]

        return y, cells, genes, x, input_dim
        
        
    def __getitem__(self, item):
        
        x_corrupted = self.corrupt(self.x[item])
        
        return self.x[item], x_corrupted, self.class_transform_dict[self.y[item]]

    def __len__(self):
        return self.x.shape[0]
    
    def corrupt(self, x, corruption_rate=0.6):
        x_corrputed = x.copy()
        
        mask = np.random.rand(len(x)) < corruption_rate
        random_idx = np.random.randint(self.x.shape[0], size=sum(mask))
        x_corrputed[mask] = self.x[random_idx, mask]

        return x_corrputed

    @classmethod
    def concat(cls, exp1, exp2, tissue_name = None):
        new_exp = cls(np.concatenate((exp1.x, exp2.x)), np.concatenate((exp1.cells, exp2.cells)) ,np.concatenate((exp1.genes, exp2.genes)), 
                        tissue_name, np.concatenate((exp1.y, exp2.y)) )
        return new_exp

