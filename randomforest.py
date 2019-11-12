import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


data = pd.read_csv("preprocessed_complete_2006_2016.csv")
subset = data.loc[:, data.columns != 'STREET_NAME']
subset = subset.loc[:,  subset.columns != 'PROPERTY_POSTAL_CODE']
subset = subset.loc[:, subset.columns != 'Unnamed: 0']
subset = subset.loc[:, subset.columns != 'PID']
subset = subset.loc[:, subset.columns != 'CURRENT_LAND_VALUE_x']
subset = subset.loc[:, subset.columns != 'CURRENT_IMPROVEMENT_VALUE_x']
subset = subset.loc[:, subset.columns != 'CURRENT_LAND_VALUE_y']
subset = subset.loc[:, subset.columns != 'CURRENT_IMPROVEMENT_VALUE_y']
subset = subset.loc[:, subset.columns != 'REPORT_YEAR_x']
subset = subset.loc[:, subset.columns != 'REPORT_YEAR_y']
subset = subset.loc[:, subset.columns != 'STREET_NAME']
subset = subset.loc[:, subset.columns != 'TAX_ASSESSMENT_YEAR']
subset = subset.loc[:, subset.columns != 'PREVIOUS_IMPROVEMENT_VALUE']
subset = subset.loc[:, subset.columns != 'PREVIOUS_LAND_VALUE']
# subset = subset.loc[:, subset.columns != 'CURRENT_IMPROVEMENT_VALUE_DELTA']
subset = subset.loc[:, subset.columns != 'TAX_LEVY_x']
subset = subset.loc[:, subset.columns != 'TAX_LEVY_y']
subset = subset.loc[:, subset.columns != 'YEAR_BUILT']
subset = subset.loc[:, subset.columns != 'BIG_IMPROVEMENT_YEAR']
subset = subset.loc[:, subset.columns != ' Total - Age groups and average age of the population - 100% data ']
subset = subset.loc[:, subset.columns != ' Total population 15 years and over by presence of children and labour force activity ']
subset = subset.loc[:, subset.columns != 'family income']
subset = subset.loc[:, subset.columns != 'total martial status']
subset = subset.loc[:, subset.columns != 'LEGAL_TYPE_STRATA']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_Multiple Family Dwelling']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_Two Family Dwelling']
subset = subset.loc[:, subset.columns != '  0 to 4 years']
subset = subset.loc[:, subset.columns != '  5 to 9 years']
subset = subset.loc[:, subset.columns != '  10 to 14 years']
subset = subset.loc[:, subset.columns != '  15 to 19 years']
subset = subset.loc[:, subset.columns != '  70 to 74 years']
subset = subset.loc[:, subset.columns != '  75 to 79 years']
subset = subset.loc[:, subset.columns != '  80 to 84 years']
subset = subset.loc[:, subset.columns != '  85 to 89 years']
subset = subset.loc[:, subset.columns != '  90 to 94 years']
subset = subset.loc[:, subset.columns != '  90 to 94 years']
subset = subset.loc[:, subset.columns != '  95 to 99 years']
subset = subset.loc[:, subset.columns != '  60 to 65 years']
subset = subset.loc[:, subset.columns != '  45 to 49 years']
subset = subset.loc[:, subset.columns != '  65 to 69 years']
subset = subset.loc[:, subset.columns != '  60 to 64 years']
subset = subset.loc[:, subset.columns != '  40 to 44 years']
subset = subset.loc[:, subset.columns != '  100 years and over']
subset = subset.loc[:, subset.columns != '    Married']
subset = subset.loc[:, subset.columns != '    Living common law']
subset = subset.loc[:, subset.columns != '    Never married']
subset = subset.loc[:, subset.columns != '   Divorced ']
subset = subset.loc[:, subset.columns != '$100000 and over']
subset = subset.loc[:, subset.columns != 'median income']
subset = subset.loc[:, subset.columns != 'REGION_West Point Grey']
subset = subset.loc[:, subset.columns != 'REGION_Kerrisdale']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_Two Family Dwelling']
subset = subset.dropna(axis=0, how='any', inplace=False)

train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_in = subset.drop('CURRENT_LAND_VALUE_DELTA', axis=1, inplace=False)
data_out = subset.loc[:, 'CURRENT_LAND_VALUE_DELTA']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]

# really basic parameters, didnt specify max or max leaf nodes, chose 1000 for n_estimators but the default is 10, got mae of 433k
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(training_data_in, training_data_out)
price_pred =rf.predict(test_data_in)

lst = []
for i in range(len(price_pred)):
    tmp = abs(test_data_out.values[i] - price_pred[i])
    lst.append(tmp)
mae = np.mean(lst)

print('Mean Absolute Error = ', mae)