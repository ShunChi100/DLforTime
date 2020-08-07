import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import plotly.graph_objects as go

    
class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x_array = x
        self.y_array = y
        
    def __len__(self):
        return len(self.y_array)
    
    def __getitem__(self, index):
        X = self.x_array[index, :]
        y = self.y_array[index]
        
        return X, y 

class StockData:
    def __init__(self, symbol, dates, transformation, data_path="../Data_Folder/Stocks", features=['Close']):
        self.symbol = symbol.lower()
        self.dates = dates
        self.transformation = transformation
        self.data_path = data_path
        self.features = features
        
        self.data_raw = self._load_data()
        self.data = self._build_data()
        self.y_true = None
        self.y_pred = None
    
    def _load_data(self):
        stocks_raw = {}
        df_dates = pd.DataFrame(index=self.dates)
        data_raw = pd.read_csv(f"{self.data_path}/{self.symbol}.us.txt", parse_dates=True, index_col=0)
        data_raw = df_dates.join(data_raw, how='inner')
        data_raw = data_raw[self.features].fillna(method='ffill')
        return data_raw
    
    def _build_data(self):
        data = self.transformation.fit_transform(self.data_raw)
        data_names = ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']
        return {k:v.astype(np.float32) for k,v in zip(data_names, data) }
    
    def inference(self, model):
        
        data_split = [['x_train', 'y_train'], ['x_val', 'y_val'], ['x_test', 'y_test']]

        y_pred = {}
        model.eval()
        device = next(model.parameters()).device
        for split in data_split:
            x_split = torch.from_numpy(self.data[split[0]].astype(np.float32)).to(device)
            y_pred[split[1][2:]] = model(x_split).detach().to("cpu").numpy()
#         for key in y_pred:
#             y_pred[key] = self.transformation.inverse_transform(y_pred[key]).flatten()
        self.y_pred = y_pred
        
        self.y_true = {}
        for key in y_pred:
            self.y_true[key] = self.data["y_"+key].flatten()
        
        look_back = self.transformation.look_back
        self.data_raw['y_pred']=np.nan
        self.data_raw['y_pred_model_output']=np.nan
        self.data_raw['y_true']=np.nan
        self.data_raw['dataset']=np.nan
        
        start_idx = look_back
        end_idx = len(self.y_pred['train'])+look_back
        self.data_raw['y_pred_model_output'].iloc[start_idx:end_idx] = self.y_pred['train'].flatten()
        self.data_raw['y_true'].iloc[start_idx:end_idx] = self.y_true['train'].flatten()
        self.data_raw['dataset'].iloc[start_idx:end_idx] = 'train'
        
        start_idx = len(self.y_pred['train'])+look_back
        end_idx = len(self.y_pred['train'])+look_back + len(self.y_pred['val'])
        self.data_raw['y_pred_model_output'].iloc[start_idx:end_idx] = self.y_pred['val'].flatten()
        self.data_raw['y_true'].iloc[start_idx:end_idx] = self.y_true['val'].flatten()
        self.data_raw['dataset'].iloc[start_idx:end_idx] = 'val'
        
        start_idx = len(self.y_pred['train'])+look_back + len(self.y_pred['val'])

        self.data_raw['y_pred_model_output'].iloc[start_idx:] = self.y_pred['test'].flatten()
        self.data_raw['y_true'].iloc[start_idx:] = self.y_true['test'].flatten()
        self.data_raw['dataset'].iloc[start_idx:] = 'test'
        
        self.data_raw = self.transformation.inverse_transform(self.data_raw)

        
    def plot(self):

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data_raw.index,
            y=self.data_raw['Close'],
            name = "y_true",
            line=dict(color='black', width=2))
            
        )
        fig.add_trace(go.Scatter(
            x=self.data_raw.loc[self.data_raw['dataset']=='train'].index,
            y=self.data_raw.loc[self.data_raw['dataset']=='train']['y_pred'],
            name = "y_pred_train",
            
        ))
        fig.add_trace(go.Scatter(
            x=self.data_raw.loc[self.data_raw['dataset']=='val'].index,
            y=self.data_raw.loc[self.data_raw['dataset']=='val']['y_pred'],
            name = "y_pred_val",
            
        ))
        fig.add_trace(go.Scatter(
            x=self.data_raw.loc[self.data_raw['dataset']=='test'].index,
            y=self.data_raw.loc[self.data_raw['dataset']=='test']['y_pred'],
            name = "y_pred_test",
            
        ))
  
        fig.show()

        
class StockData_ensemble:
    def __init__(self, symbols, dates, transformation, max_num=100000000000, data_path="../Data_Folder/Stocks"):
        self.symbols = [symbol.lower() for symbol in symbols]
        self.dates = dates
        self.transformation = transformation
        self.symbols_used = []
        self.max_num = max_num
        self.data_path = data_path
        
        self.data = self._build_data()
        
    
    def _build_data(self):
        data = {}
        for symbol in self.symbols:
            try:
                single_stock = StockData(symbol, self.dates, transformation=self.transformation, data_path = self.data_path)
            except:
                continue
            if np.mean(single_stock.data_raw['Close'].iloc[0:100]) < 2:
                continue
            self.symbols_used.append(symbol)
            if len(self.symbols_used) % 100 == 0:
                print(symbol, end="|")
            if len(data) < 1:
                data = single_stock.data
            else:
                for key in data:
                    data[key] = np.concatenate([data[key], single_stock.data[key]], axis=0)
                    
            if len(self.symbols_used) >= self.max_num:
                break

        return data
