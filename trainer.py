import torch
from model.net import Net
from data.utils import init_data_loaders
import pandas as pd
from data import dataset
from model.loss import NTXent
from sklearn.metrics import accuracy_score, f1_score


class Trainer():

    def __init__(self, df_dataset: pd.DataFrame, configs):
        self.configs = configs
        self.dataset = self.init_dataset(df_dataset)
        self.pretrain_loader, self.finetune_loader, self.val_loader, self.test_loader = init_data_loaders(self.dataset, configs.batch_size)
        self.model = Net(self.dataset.input_dim, self.dataset.n_classes, configs)
        if self.configs.cuda:
            self.model = self.model.cuda()

    def init_dataset(self, df_dataset):
        if self.configs.experiment == 'single_cell':
            return dataset.SingleCellDataset(df_dataset)
        else:
            return dataset.TabularDataset(df_dataset)
        
    def run(self):
        #self.pretrain()
        self.finetune()
        self.test()


    def pretrain(self):
        if self.configs.verbose:
            print("Pretraining Starts")
        self.model.train()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.configs.pretrain_lr)
        criterion = NTXent()

        es_patience = 0
        prev_val_loss = 1000
        for iteration in range(self.configs.n_epochs):
            epoch_loss = 0
            for x, x_corrupted, y in self.pretrain_loader:
                if self.configs.cuda:
                    x, x_corrupted, y = x.cuda(), x_corrupted.cuda(), y.cuda()
                _, z, _, _ = self.model(x)
                _, z_corrupted, _ , _ = self.model(x_corrupted)

                loss = criterion(z, z_corrupted)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            val_loss, _ = self.evaluate()
            if self.configs.verbose:
                print(f"\tVal SS Loss: {val_loss[0]:.3f}")
            if  prev_val_loss <= val_loss[0]:
                if es_patience+1 < self.configs.early_stoping_patience:
                    print("Patience++")
                    es_patience += 1
                else:
                    if self.configs.verbose:
                        print(f"Early stopping at iteration {iteration}")
                    break
            else:
                es_patience = 0
            prev_val_loss = val_loss[0]
            
            print(f"Loss: {loss.item():.3f}")


    def finetune(self):
        if self.configs.verbose:
            print("finetuning Starts")
        self.model.train()
        optimizer = torch.optim.Adam(params=self.model.parameters(),lr=self.configs.pretrain_lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        es_patience = 0
        prev_val_acc = 0
        for iteration in range(self.configs.n_epochs):
            epoch_loss = 0
            for x, _, y in self.finetune_loader:
                if self.configs.cuda:
                    x, y = x.cuda(), y.cuda()
                _, _ , pred_raw, _ = self.model(x)
                loss = criterion(pred_raw, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            _, metrics = self.evaluate()
            if self.configs.verbose:
                print(f"Val accuracy: {metrics['accuracy']:.3f}")
            if metrics['accuracy'] - prev_val_acc < self.configs.finetuning_early_stoping_threshold:
                if es_patience+1 < self.configs.early_stoping_patience:
                    es_patience += 1
                else:
                    if self.configs.verbose:
                        print(f"Early stopping at iteration {iteration}")
                    # break
            else:
                es_patience = 0
            prev_val_acc = metrics['accuracy']        
            print(f"Loss: {epoch_loss:.3f}")
        

    def evaluate(self):
        self.model.eval()
        preds = []
        targets = []
        ce_loss = torch.nn.CrossEntropyLoss()
        validation_ce_loss = 0
        validation_ss_loss = 0
        contrastive_loss = NTXent()
        for x, x_corrupted, y in self.val_loader:
            if self.configs.cuda:
                x, x_corrupted, y = x.cuda(), x_corrupted.cuda(), y.cuda()
            _, z, pred_raw , pred = self.model(x)
            _, z_corrupted, ـ , ـ = self.model(x_corrupted)
            validation_ce_loss += ce_loss(pred_raw, y).item()
            validation_ss_loss += contrastive_loss(z, z_corrupted).item()
            preds.append(pred.cpu())
            targets.append(y.cpu())
            
        preds = torch.concatenate(preds, dim=0)
        targets = torch.concatenate(targets, dim=0)
        pred_classes = torch.argmax(preds, dim=1)
        
        accuracy = accuracy_score(targets, pred_classes)
        f1 = None#f1_score(targets, pred_classes)
        metrics = {'accuracy': accuracy, 'f1-score':f1}
        # print(f"Accuracy: {accuracy:.3f}")
        # print(f"F1-Score: {f1:.3f}")
        self.model.train()
        return (validation_ss_loss, validation_ce_loss), metrics
            
    def test(self):
        self.model.eval()
        preds = []
        targets = []
        criterion = torch.nn.CrossEntropyLoss()
       
        for x, _, y in self.test_loader:
            if self.configs.cuda:
                x, y = x.cuda(), y.cuda()
            _, _, _ , pred = self.model(x)
            preds.append(pred.cpu())
            targets.append(y.cpu())
            
        preds = torch.concatenate(preds, dim=0)
        targets = torch.concatenate(targets, dim=0)
        pred_classes = torch.argmax(preds, dim=1)
        accuracy = accuracy_score(targets, pred_classes)
        f1 = None# f1_score(targets, pred_classes)
        metrics = {'accuracy': accuracy, 'f1-score':f1}
        print(f"Accuracy: {accuracy:.3f}")
        #print(f"F1-Score: {f1:.3f}")
       
        return metrics