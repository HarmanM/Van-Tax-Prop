import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import r2_score


def poly_kfoldCV(x, y, K, p):
    arr = ['CURRENT_IMPROVEMENT_VALUE_DELTA', 'LEGAL_TYPE_STRATA', 'LEGAL_TYPE_LAND', 'LEGAL_TYPE_OTHER',
           'ZONE_CATEGORY_Commercial',
           'ZONE_CATEGORY_One Family Dwelling', 'ZONE_CATEGORY_Light Industrial',
           'ZONE_CATEGORY_Comprehensive Development',
           'REGION_Shaughnessy', 'REGION_Grandview-Woodland', '   $10000 to $19999 ', 'REGION_Sunset',
           'ZONE_CATEGORY_Two Family Dwelling',
           '   $20000 to $29999 ', 'REGION_West Point Grey']
    arr_transf = arr[:p]
    #print(arr[:p])
    subset_in = x.loc[:, arr_transf]
    subset_in = np.array_split(subset_in, K)
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


data = pd.read_csv("demo.csv")
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

arr = ['CURRENT_IMPROVEMENT_VALUE_DELTA', 'LEGAL_TYPE_STRATA', 'LEGAL_TYPE_LAND', 'LEGAL_TYPE_OTHER', 'ZONE_CATEGORY_Commercial',
       'ZONE_CATEGORY_One Family Dwelling', 'ZONE_CATEGORY_Light Industrial', 'ZONE_CATEGORY_Comprehensive Development',
       'REGION_Shaughnessy', 'REGION_Grandview-Woodland', '   $10000 to $19999 ', 'REGION_Sunset', 'ZONE_CATEGORY_Two Family Dwelling',
       '   $20000 to $29999 ', 'REGION_West Point Grey']

train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

shuffled_indices = list(range(num_rows))
#random.seed(42)

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

train_data = subset.iloc[train_indices, :]
test_data = subset.iloc[test_indices, :]

training_data_in = train_data.loc[:, ['CURRENT_IMPROVEMENT_VALUE_DELTA', 'LEGAL_TYPE_STRATA', 'LEGAL_TYPE_LAND', 'LEGAL_TYPE_OTHER',
           'ZONE_CATEGORY_Commercial',
           'ZONE_CATEGORY_One Family Dwelling', 'ZONE_CATEGORY_Light Industrial',
           'ZONE_CATEGORY_Comprehensive Development', 'CURRENT_LAND_VALUE_DELTA']]

test_data_in = test_data.loc[:, ['CURRENT_IMPROVEMENT_VALUE_DELTA', 'LEGAL_TYPE_STRATA', 'LEGAL_TYPE_LAND', 'LEGAL_TYPE_OTHER',
           'ZONE_CATEGORY_Commercial',
           'ZONE_CATEGORY_One Family Dwelling', 'ZONE_CATEGORY_Light Industrial',
           'ZONE_CATEGORY_Comprehensive Development', 'CURRENT_LAND_VALUE_DELTA']]

training_data_in = training_data_in.drop('CURRENT_LAND_VALUE_DELTA', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'CURRENT_LAND_VALUE_DELTA']

test_data_in = test_data_in.drop('CURRENT_LAND_VALUE_DELTA', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'CURRENT_LAND_VALUE_DELTA']


#pt3_train_arr = []
#pt3_valid_arr = []
#for i in range(16):
#    if i > 1:
#      kfold_result = poly_kfoldCV(training_data_in, training_data_out, i, 2)
#      pt3_train_arr.append(kfold_result[1])
#      pt3_valid_arr.append(kfold_result[0])
#      print(i, "training: ", kfold_result[1], "cv: ", kfold_result[0])

#plt.plot(range(1, 15), pt3_train_arr)
#plt.plot(range(1, 15), pt3_valid_arr)
#plt.suptitle("Learning curve plot for feature # vs. mae")
#plt.xlabel("Number of features = ")
#plt.ylabel("Error")
#plt.legend(['y = train_error', 'y = cv_error'], loc='upper left')
#plt.show()

poly = PolynomialFeatures(degree=1)

x_transf = poly.fit_transform(training_data_in)
x_test_transf = poly.fit_transform(test_data_in)

lin_reg = LinearRegression()
lin_reg.fit(x_transf, training_data_out)
price_pred = lin_reg.predict(x_test_transf)


lst = []
for i in range(len(price_pred)):
    tmp = abs(test_data_out.values[i] - price_pred[i])
    lst.append(tmp)
mae = np.mean(lst)

print('Mean Absolute Error = ', mae)

r2 = r2_score(list(test_data_out), price_pred)
print("r2 score: ", r2)

print(np.array(test_data['PCOORD'][0:5]))

export = pd.DataFrame(columns=['Predicted', 'Actual', 'Difference'])
export['Predicted'] = price_pred[0:5]
export['Actual'] = test_data_out.values[0:5]
export['Difference'] = np.subtract(price_pred[0:5], test_data_out.values[0:5])
export['Pcoord'] = np.array(test_data['PCOORD'][0:5])
export['Geom'] = np.array(test_data['Geom'][0:5])
export.to_csv('linear_regression.csv')
print(export)