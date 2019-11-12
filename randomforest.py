import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def forest_kfoldCV(x, y, K, n):
    # really basic parameters, didnt specify max or max leaf nodes, chose 1000 for n_estimators but the default is 10, got mae of 433k
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_features=n)
    subset_in = np.array_split(x, K)
    subset_out = np.array_split(y, K)
    cut_off_x = len(subset_in[K - 1])
    cut_off_y = len(subset_out[K - 1])
    t_mae = []
    c_mae = []
    for i in range(len(subset_in)):
        subset_in[i] = subset_in[i][0:cut_off_x]
    for i in range(len(subset_out)):
        subset_out[i] = subset_out[i][0:cut_off_y]
    for i in range(K):
        validation_setin = np.array(subset_in[i])
        validation_setout = np.array(subset_out[i])
        if (i == 0):
            training_setin = np.concatenate(subset_in[1:])
            training_setout = np.concatenate(subset_out[1:])
        elif (i == K - 1):
            training_setin = np.concatenate(subset_in[0:i])
            training_setout= np.concatenate(subset_out[0:i])
        else:
            training_setin = np.concatenate(subset_in[0:i] + subset_in[i + 1:])
            training_setout = np.concatenate(subset_out[0:i] + subset_out[i + 1:])

        rf.fit(training_setin, training_setout)

        y_pred_val = rf.predict(validation_setin)
        y_pred_train = rf.predict(training_setin)

        lst = []
        for n in range(len(y_pred_val)):
            tmp = abs(validation_setout[n] - y_pred_val[n])
            lst.append(tmp)
        c_mae.append(np.mean(lst))
        print("cv", np.mean(lst))
        lst = []
        for i in range(len(y_pred_train)):
            tmp = abs(training_setout[i] - y_pred_train[i])
            lst.append(tmp)
        t_mae.append(np.mean(lst))

        print("training", np.mean(lst))

    train_error = np.mean(t_mae)
    cv_error = np.mean(c_mae)
    return cv_error, train_error

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

pt3_train_arr = []
pt3_valid_arr = []
for i in range(16):
    kfold_result = forest_kfoldCV(training_data_in, training_data_out, 5, i + 1)
    pt3_train_arr.append(kfold_result[1])
    pt3_valid_arr.append(kfold_result[0])
    print(i, "training: ", kfold_result[1], "cv: ", kfold_result[0])

plt.plot(range(1, 16), pt3_train_arr)
plt.plot(range(1, 16), pt3_valid_arr)
plt.suptitle("Learning curve plot for num_features vs. mae")
plt.xlabel("num_features = ")
plt.ylabel("Error")
plt.legend(['y = train_error', 'y = cv_error'], loc='upper left')
plt.show()