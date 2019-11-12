import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt

def lasso_kfoldCV(x, y, p, k):
    cv_error_arr = [None] * k
    train_error_arr = [None] * k

    #set the lamda value
    lasso = linear_model.Lasso(alpha=p)
    #split x and y into k folds
    inputs = np.array_split(np.array(x), k)
    output = np.array_split(np.array(y), k)

    #performs cross validation and finds mae of cv, trainig err
    for i in range(k):
        #get the training set and training outputs
        train_input = np.concatenate(inputs[:i] + inputs[i+1:])
        train_output = np.concatenate(output[:i] + output[i+1:])

        #fit ridge model with training set
        lasso.fit(train_input, train_output)

        #predict using validation set
        validation_pred = lasso.predict(inputs[i])

        #predict using training outputs
        train_pred = lasso.predict(train_input)

        #store the mae of the training err and cv err
        cv_error_arr[i] = np.mean(np.absolute(np.subtract(output[i], validation_pred)))
        train_error_arr[i] = np.mean(np.absolute(np.subtract(train_output, train_pred)))

    #return averages of mae
    train_err = sum(train_error_arr)/k
    cv_err = sum(cv_error_arr)/k

    return train_err, cv_err


data = pd.read_csv("preprocessed_complete_2006_2016_v2.csv")
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

# subset = subset.loc[:, subset.columns != 'CURRENT_LAND_VALUE_DELTA']
subset = subset.loc[:, subset.columns != 'LEGAL_TYPE_STRATA']
subset = subset.loc[:, subset.columns != 'LEGAL_TYPE_LAND']
subset = subset.loc[:, subset.columns != 'LEGAL_TYPE_OTHER']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_One Family Dwelling']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_Multiple Family Dwelling']
subset = subset.loc[:, subset.columns != 'ZONE_CATEGORY_Two Family Dwelling']
subset = subset.loc[:, subset.columns != 'REGION_South Cambie']
subset = subset.loc[:, subset.columns != 'TAX_LEVY_x']
subset = subset.loc[:, subset.columns != 'TAX_LEVY_y']
subset = subset.loc[:, subset.columns != 'YEAR_BUILT']
subset = subset.loc[:, subset.columns != 'BIG_IMPROVEMENT_YEAR']
subset = subset.loc[:, subset.columns != ' Total - Age groups and average age of the population - 100% data ']


# cols = [col for col in data.columns if col in ['CURRENT_LAND_VALUE_DELTA', 'REGION_Kitsilano','ZONE_CATEGORY_Comprehensive Development', 'REGION_Killarney', 'REGION_Dunbar-Southlands', 'ZONE_CATEGORY_Industrial', '   $50000 to $59999 '
#     , 'REGION_Grandview-Woodland', 'ZONE_CATEGORY_Commercial', 'REGION_Oakridge', 'REGION_Riley', 'REGION_Kerrisdale', 'YEAR_BUILT', 'REGION_West Point Grey', 'ZONE_CATEGORY_Light Industrial', '   $20000 to $29999 '
#     , 'REGION_Shaughnessy', 'BIG_IMPROVEMENT_YEAR', 'CURRENT_IMPROVEMENT_VALUE_DELTA']]
# subset = data[cols]
subset = subset.dropna(axis=0, how='any', inplace=False)

print(subset)
data_feature = subset.drop(['CURRENT_LAND_VALUE_DELTA'], axis=1, inplace=False)
scaler = StandardScaler()

#standardize data excluding target_D
standardize_data = pd.DataFrame(scaler.fit_transform(data_feature), columns = data_feature.columns)
train_ratio = .3
num_rows = subset.shape[0]
indices = list(range(num_rows))

train_set_size = int(num_rows * train_ratio)

# training set: take the first 'train_set_size' rows
train_indices = indices[:train_set_size]
# test set: take the remaining rows
test_indices = indices[train_set_size:]

# create training and testing features
train_features = standardize_data.iloc[train_indices, :]
test_features = standardize_data.iloc[test_indices, :]

# create training label
train_data_labels = subset.iloc[train_indices, :]
train_labels = train_data_labels.loc[:,'CURRENT_LAND_VALUE_DELTA']

# create test labels
test_data_labels = subset.iloc[test_indices, :]
test_labels = test_data_labels.loc[:, 'CURRENT_LAND_VALUE_DELTA']

print(train_labels)
train_err = []
cv_err = []
lam = [ 10**3.5, 10**4, 10**4.5, 10**5, 10**5.5, 10**6, 10**7, 10**8]
# train_err_result, cv_err_result = lasso_kfoldCV(train_features, train_labels, 10**-1,  5)

#find the average cv and training mae using lasso regression
for l in list(lam):
    train_err_result, cv_err_result = lasso_kfoldCV(train_features, train_labels, l,  5)
    train_err.append(train_err_result)
    cv_err.append(cv_err_result)

plt.title("Lasso Regression of MAE and λ")
plt.plot(np.log10(lam), train_err, '-b', label="training error")
plt.plot(np.log10(lam), cv_err, '-g', label="cross validation error")
plt.xticks(range(3,8))
plt.xlabel("log10(λ)")
plt.ylabel("MAE")
plt.xlim()
plt.legend()
plt.show()

lasso = linear_model.Lasso(alpha=10**4.5)
#fi5 using training features
lasso.fit(train_features, train_labels)

#predict using test features set
ridge_pred = lasso.predict(test_features)

#calculate mae using ridge regression
# mae = np.mean(np.absolute(np.subtract(ridge_pred, test_labels)))
# print('\nMean Absolute Error when lambda is 10^-.5 = ', mae)

#finding the top 3 features
coef = dict(zip(train_features, lasso.coef_))
# coef = {x:y for x,y in coef.items() if y!=0}
top_feature = sorted(coef, key=lambda coef_key: abs(coef[coef_key]), reverse=True)[:10]
print("\nTop 10 features: ", top_feature)
print(sorted(lasso.coef_, reverse=True)[:10])
print(coef)
