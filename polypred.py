import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def poly_kfoldCV(x, y, K, p):
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

        poly = PolynomialFeatures(degree=p)

        x_transf = poly.fit_transform(training_setin)
        valid_transf = poly.fit_transform(validation_setin)

        lin_reg = LinearRegression()
        lin_reg.fit(x_transf, training_setout)
        y_pred_val = lin_reg.predict(valid_transf)
        y_pred_train = lin_reg.predict(x_transf)

        lst = []
        for n in range(len(y_pred_val)):
            tmp = abs(validation_setout[n] - y_pred_val[n])
            lst.append(tmp)
        c_mae.append(np.mean(lst))

        lst = []
        for i in range(len(y_pred_train)):
            tmp = abs(training_setout[i] - y_pred_train[i])
            lst.append(tmp)
        t_mae.append(np.mean(lst))


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

subset = subset.loc[:, subset.columns != 'CURRENT_LAND_VALUE_DELTA']
subset = subset.loc[:, subset.columns != 'LEGAL_TYPE_STRATA']
subset = subset.loc[:, subset.columns != 'LEGAL_TYPE_LAND']
subset = subset.loc[:, subset.columns != 'LEGAL_TYPE_OTHER']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_One Family Dwelling']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_Multiple Family Dwelling']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_Two Family Dwelling']
subset = subset.loc[:, subset.columns != 'REGION_South Cambie']
subset = subset.dropna(axis=0, how='any', inplace=False)

train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_in = subset.drop('CURRENT_IMPROVEMENT_VALUE_DELTA', axis=1, inplace=False)
data_out = subset.loc[:, 'CURRENT_IMPROVEMENT_VALUE_DELTA']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

training_data_in = StandardScaler(with_mean=True, with_std=True).fit_transform(training_data_in)

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]


pt3_train_arr = []
pt3_valid_arr = []
for i in range(16):
    if i > 1:
        print(i)
        kfold_result = poly_kfoldCV(training_data_in, training_data_out, i + 1, 2)
        pt3_train_arr.append(kfold_result[1])
        pt3_valid_arr.append(kfold_result[0])

plt.plot(range(2, 16), pt3_train_arr)
plt.plot(range(2, 16), pt3_valid_arr)
plt.suptitle("Learning curve plot for p vs. mae")
plt.xlabel("degree of fitting polynomial")
plt.ylabel("Error")
plt.legend(['y = train_error', 'y = cv_error'], loc='upper left')
plt.show()