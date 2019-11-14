import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score




def forest_kfoldCV(x, y, K, n):
    # really basic parameters, didnt specify max or max leaf nodes, chose 1000 for n_estimators but the default is 10, got mae of 433k
    rf = RandomForestRegressor(n_estimators=20, random_state=42, max_depth=12, min_samples_split=2)
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
#       print(pd.Series(rf.feature_importances_, index=training_data_in.columns))

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

train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_in = subset.drop('CURRENT_LAND_VALUE_DELTA', axis=1, inplace=False)
data_out = subset.loc[:, 'CURRENT_LAND_VALUE_DELTA']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]
temp_test_in = test_data_in

training_data_in = training_data_in.loc[:, ['CURRENT_IMPROVEMENT_VALUE_DELTA', 'LEGAL_TYPE_STRATA',
                           'LEGAL_TYPE_LAND', 'LEGAL_TYPE_OTHER', 'ZONE_CATEGORY_Commercial',
                           'ZONE_CATEGORY_One Family Dwelling', 'ZONE_CATEGORY_Light Industrial',
                           'ZONE_CATEGORY_Comprehensive Development', 'REGION_Shaughnessy',
                           'REGION_Grandview-Woodland', '   $10000 to $19999 ', 'REGION_Sunset',
                           'ZONE_CATEGORY_Two Family Dwelling', '   $20000 to $29999 ', 'REGION_West Point Grey']]

test_data_in = test_data_in.loc[:, ['CURRENT_IMPROVEMENT_VALUE_DELTA', 'LEGAL_TYPE_STRATA',
                           'LEGAL_TYPE_LAND', 'LEGAL_TYPE_OTHER', 'ZONE_CATEGORY_Commercial',
                           'ZONE_CATEGORY_One Family Dwelling', 'ZONE_CATEGORY_Light Industrial',
                           'ZONE_CATEGORY_Comprehensive Development', 'REGION_Shaughnessy',
                           'REGION_Grandview-Woodland', '   $10000 to $19999 ', 'REGION_Sunset',
                           'ZONE_CATEGORY_Two Family Dwelling', '   $20000 to $29999 ', 'REGION_West Point Grey']]


#pt3_train_arr = []
#pt3_valid_arr = []
#for i in range(10):
#    kfold_result = forest_kfoldCV(training_data_in, training_data_out, i + 2, 0)
#    pt3_train_arr.append(kfold_result[1])
#    pt3_valid_arr.append(kfold_result[0])
#    print(i, "training: ", kfold_result[1], "cv: ", kfold_result[0])

#plt.plot(range(2, 12), pt3_train_arr)
#plt.plot(range(2, 12), pt3_valid_arr)
#plt.suptitle("Min samples to split vs. mae")
#plt.xlabel("Min samples to split ")
#plt.ylabel("Error")
#plt.legend(['y = train_error', 'y = cv_error'], loc='upper left')
#plt.show()

#print(training_data_in.head(20).to_string())

rf2 = RandomForestRegressor(n_estimators=20, random_state=42, max_depth=12, min_samples_split=2)
rf2.fit(training_data_in, training_data_out)
y_pred_val = rf2.predict(test_data_in)

# print(y_pred_val[0])
# print(list(test_data_out)[0])

lst = []
for n in range(len(y_pred_val)):
    tmp = abs(list(test_data_out)[n] - y_pred_val[n])
    lst.append(tmp)
mae = np.mean(lst)

# print("Computed mae: ", mae)
# mape = 100 * (mae / list(test_data_in))
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')

r2 = r2_score(list(test_data_out), y_pred_val)
print(r2)

print(temp_test_in.head(20).to_string())

export = pd.DataFrame(columns=['Predicted', 'Actual', 'Difference'])
export['Predicted'] = y_pred_val[0:5]
export['Actual'] = test_data_out.values[0:5]
export['Difference'] = np.subtract(y_pred_val[0:5], test_data_out.values[0:5])
export['Pcoord'] = np.array(temp_test_in['PCOORD'][0:5])
export['Geom'] = np.array(temp_test_in['Geom'][0:5])
export.to_csv('random_forest.csv')
print(export)