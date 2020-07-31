import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule


class Model(LightningModule):
    def __init__(self, model):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        params = {'batch_size': 256,
                  'shuffle': True,
                  'num_workers': 6}
        
        train_dataset = StockDataset(data_all.data['x_train'], data_all.data['y_train'])
        
        return DataLoader(train_dataset, **params)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
#     def mse_loss(self):
#         loss = torch.nn.MSELoss(size_average=True)
#         return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        params = {'batch_size': 256,
                  'shuffle': False,
                  'num_workers': 6}
        
        val_dataset = StockDataset(data_all.data['x_val'], data_all.data['y_val'])
        return DataLoader(val_dataset, **params)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self):
        params = {'batch_size': 256,
                  'shuffle': False,
                  'num_workers': 6}
        
        test_dataset = StockDataset(data_all.data['x_test'], data_all.data['y_test'])
        return DataLoader(test_dataset, **params)