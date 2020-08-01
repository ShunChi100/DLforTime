import sklearn.metrics as metrics
import pandas as pd
import numpy as np
    
class Evaluator:
    def __init__(self):
        self.metrics = pd.Series()
        #self.evaluate()
    
    def evaluate(self, y_pred, y_true):
        df = self._make_df(y_pred, y_true)
        
        self.metrics.loc['explained variance'] = metrics.explained_variance_score(y_true, y_pred)
        self.metrics.loc['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
        self.metrics.loc['MSE'] = metrics.mean_squared_error(y_true, y_pred)
        self.metrics.loc['MedAE'] = metrics.median_absolute_error(y_true, y_pred)
        self.metrics.loc['RSQ'] = metrics.r2_score(y_true, y_pred)
        
        # building block metrics
        self.metrics.loc['accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
        self.metrics.loc['edge'] = df.result.mean()
        self.metrics.loc['noise'] = df.y_pred.diff().abs().mean()

        # derived metrics
        self.metrics.loc['y_true_chg'] = df.y_true.abs().mean()
        self.metrics.loc['y_pred_chg'] = df.y_pred.abs().mean()
        self.metrics.loc['prediction_calibration'] = self.metrics.loc['y_pred_chg']/self.metrics.loc['y_true_chg']
        self.metrics.loc['capture_ratio'] = self.metrics.loc['edge']/self.metrics.loc['y_true_chg']*100

        # metrics for a subset of predictions
        self.metrics.loc['edge_long'] = df[df.sign_pred == 1].result.mean()  - df.y_true.mean()
        self.metrics.loc['edge_short'] = df[df.sign_pred == -1].result.mean()  - df.y_true.mean()

        self.metrics.loc['edge_win'] = df[df.is_correct == 1].result.mean()  - df.y_true.mean()
        self.metrics.loc['edge_lose'] = df[df.is_incorrect == 1].result.mean()  - df.y_true.mean()


        
    def _make_df(self, y_pred, y_true):
        
        y_pred.name = 'y_pred'
        y_true.name = 'y_true'

        df = pd.DataFrame({"y_pred": y_pred.pct_change(-1)*100, "y_true": y_true.pct_change(-1)*100})

        df['sign_pred'] = df.y_pred.apply(np.sign)
        df['sign_true'] = df.y_true.apply(np.sign)
        df['is_correct'] = 0
        df.loc[df.sign_pred * df.sign_true > 0 ,'is_correct'] = 1 # only registers 1 when prediction was made AND it was correct
        df['is_incorrect'] = 0
        df.loc[df.sign_pred * df.sign_true < 0,'is_incorrect'] = 1 # only registers 1 when prediction was made AND it was wrong
        df['is_predicted'] = df.is_correct + df.is_incorrect
        df['result'] = df.sign_pred * df.y_true 
        return df
    
def get_buy_sell_points(y_test_pred, y_test, thresh=0.01):
    buys = []
    sells = []

    counter = 0
    for predicted, price_today in zip(y_test_pred[1:], y_test[:-1]):
        delta = predicted - price_today
        # print(delta)
        if delta > thresh*price_today:
            buys.append((counter, price_today[0]))
        elif delta < -thresh*price_today:
            sells.append((counter, price_today[0]))
        counter += 1
    return buys, sells


def compute_earnings(buys, sells):
    purchase_amt = 10
    stock = 0
    balance = 0
    while len(buys) > 0 and len(sells) > 0:
        if buys[0][0] < sells[0][0]:
            # time to buy $10 worth of stock
            balance -= purchase_amt
            stock += purchase_amt / buys[0][1]
            buys.pop(0)
        else:
            # time to sell all of our stock
            balance += stock * sells[0][1]
            stock = 0
            sells.pop(0)
    print(balance)