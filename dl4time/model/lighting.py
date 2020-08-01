import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.lightning import LightningModule

from dl4time.data.databuilder import StockDataset


class Model(LightningModule):
    def __init__(self, model, data_container, dataloader_params=None):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = model

        self.data_container = data_container

        if dataloader_params is None:
            self.dataloader_params = {'batch_size': 256,
                                      'num_workers': 6}
        else:
            self.dataloader_params = dataloader_params

    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        
        train_dataset = StockDataset(self.data_container .data['x_train'], self.data_container .data['y_train'])
        
        return DataLoader(train_dataset, shuffle=True, **self.dataloader_params)
    
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
        val_dataset = StockDataset(self.data_container .data['x_val'], self.data_container .data['y_val'])
        return DataLoader(val_dataset, shuffle=False, **self.dataloader_params)
        
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
        test_dataset = StockDataset(self.data_container .data['x_test'], self.data_container .data['y_test'])
        return DataLoader(test_dataset, shuffle=False, **self.dataloader_params)