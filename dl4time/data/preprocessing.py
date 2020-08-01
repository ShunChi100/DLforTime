from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class MinMaxTransformation:
    def __init__(self, look_back=60):
        self.scaler = MinMaxScaler()
        self.look_back = look_back
     
    def fit_transform(self, data_raw):
        data_scaled = self.scaler.fit_transform(data_raw)
        #data_scaled = data_scaled.values

        data = []

        for index in range(len(data_scaled) - self.look_back): 
            data.append(data_scaled[index: (index + self.look_back+1)])
        data = np.array(data)
        
        val_set_size = int(np.round(0.1*data.shape[0]))
        test_set_size = int(np.round(0.2*data.shape[0]))
        train_set_size = data.shape[0] - (val_set_size + test_set_size)

        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_val = data[train_set_size:(train_set_size+val_set_size),:-1,:]
        y_val = data[train_set_size:(train_set_size+val_set_size),-1,:]

        x_test = data[(train_set_size+val_set_size):,:-1,:]
        y_test = data[(train_set_size+val_set_size):,-1,:]

        return [x_train, y_train, x_val, y_val, x_test, y_test]
    
    def inverse_transform(self, df_data_raw):
        df_data_raw['y_pred'] = self.scaler.inverse_transform(df_data_raw['y_pred_model_output'].values.reshape(-1, 1))
        return df_data_raw
    
    
class NormalizationCurrentDay:
    def __init__(self, look_back=60):
        self.look_back = look_back
     
    def fit_transform(self, data_raw):
        data_raw = data_raw['Close'].values.reshape([-1,1])
        #data_scaled = data_scaled.values

        data = []

        for index in range(len(data_raw) - self.look_back): 
            data.append(data_raw[index: (index + self.look_back+1)])
        data = np.array(data)
#         print(data_raw[0:60])
#         print(data.shape)
        data = self.scaler(data)
        
        
        val_set_size = int(np.round(0.1*data.shape[0]))
        test_set_size = int(np.round(0.2*data.shape[0]));
        train_set_size = data.shape[0] - (val_set_size + test_set_size);

        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_val = data[train_set_size:(train_set_size+val_set_size),:-1,:]
        y_val = data[train_set_size:(train_set_size+val_set_size),-1,:]

        x_test = data[(train_set_size+val_set_size):,:-1,:]
        y_test = data[(train_set_size+val_set_size):,-1,:]

        return [x_train, y_train, x_val, y_val, x_test, y_test]
    
    def inverse_transform(self, df_data_raw):
        df_data_raw['y_pred'].iloc[1:] = (df_data_raw['y_pred_model_output'].iloc[1:].values/100+1)*df_data_raw['Close'].iloc[0:-1].values
        return df_data_raw
    
    def scaler(self, data):
        for i in range(data.shape[0]):
            data[i,:,:] /= data[i, -2, 0]
        data = (data-1)*100
        return data