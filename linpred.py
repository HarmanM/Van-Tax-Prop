import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


data = pd.read_csv("new_output.csv")
subset = data.loc[:, data.columns != 'STREET_NAME']
subset = subset.loc[:,  subset.columns != 'PROPERTY_POSTAL_CODE']
subset = subset.loc[:, subset.columns != 'Unnamed: 0']

train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_in = subset.drop('CURRENT_LAND_VALUE', axis=1, inplace=False)
data_out = subset.loc[:,'CURRENT_LAND_VALUE']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]


linreg = LinearRegression()
linreg.fit(training_data_in, training_data_out)
price_pred =linreg.predict(test_data_in)

lst = []
for i in range(len(price_pred)):
    tmp = abs(test_data_out.values[i] - price_pred[i])
    lst.append(tmp)
mae = np.mean(lst)

print('Mean Absolute Error = ', mae)