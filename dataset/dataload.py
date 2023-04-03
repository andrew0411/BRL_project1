import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Dataload and split
train_df = pd.read_csv('./train.csv', encoding='cp949', indexl_col='compound')
test_df = pd.read_csv('./test.csv', encoding='cp949', indexl_col='compound')

y_tr = train_df.pop('target')
y_ts = test_df.pop('target')

x_tr, x_ts = train_df, test_df

# Delete columns with unique value

