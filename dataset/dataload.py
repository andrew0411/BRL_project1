import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def dataload():
  # Dataload and split
  train_df = pd.read_csv('./train.csv', encoding='cp949', indexl_col='compound')
  test_df = pd.read_csv('./test.csv', encoding='cp949', indexl_col='compound')

  y_tr = train_df.pop('target')
  y_ts = test_df.pop('target')

  x_tr, x_ts = train_df, test_df

  # Delete columns with unique value

  for c in x_tr.columns:
      if len(np.unique(x_tr[c])) == 1:
          repeated.append(c)
          x_tr = x_tr.drop([c], axis=1)
          
  for c in repeated:
    x_ts = x_ts.drop([c], axis=1)
    
    
  scaler = MinMaxScaler()

  x_tr_sc = scaler.fit_transform(x_tr)
  x_ts_sc = scaler.transform(x_ts)

  x_tr = pd.DataFrame(x_tr_sc, columns=x_tr.columns, index=x_tr.index)
  x_ts = pd.DataFrame(x_ts_sc, columns=x_ts.columns, index=x_ts.index)
  
  return x_tr, x_ts, y_tr, y_ts
