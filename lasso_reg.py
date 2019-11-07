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


data = pd.read_csv("new_output.csv")
subset = data.loc[:, data.columns != 'STREET_NAME']
subset = subset.loc[:,  subset.columns != 'PROPERTY_POSTAL_CODE']
subset = subset.loc[:, subset.columns != 'Unnamed: 0']

data_feature = subset.drop(['CURRENT_LAND_VALUE'], axis=1, inplace=False)
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
train_labels = train_data_labels.loc[:,'CURRENT_LAND_VALUE']

# create test labels
test_data_labels = subset.iloc[test_indices, :]
test_labels = test_data_labels.loc[:, 'CURRENT_LAND_VALUE']

train_err = []
cv_err = []
lam = [10**-2, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1, 10**-0.75, 10**-.5, 10**-.25, 10**0, 10**.25, 10**.5, 10**.75, 10**1, 10**1.25, 10**1.5, 10**1.75, 10**4]
train_err_result, cv_err_result = lasso_kfoldCV(train_features, train_labels, 10**-1,  5)

#find the average cv and training mae using lasso regression
for l in list(lam):
    train_err_result, cv_err_result = lasso_kfoldCV(train_features, train_labels, l,  5)
    train_err.append(train_err_result)
    cv_err.append(cv_err_result)

plt.title("Lasso Regression of MAE and λ")
plt.plot(np.log10(lam), train_err, '-b', label="training error")
plt.plot(np.log10(lam), cv_err, '-g', label="cross validation error")
# plt.plot(np.log10(lamda), mae_arr, '-r', label="mae from testing set")
plt.xticks(range(-2,2))
plt.xlabel("log10(λ)")
plt.ylabel("MAE")
plt.xlim()
plt.legend()
plt.show()

lasso = linear_model.Lasso(alpha=10**4)
#fi5 using training features
lasso.fit(train_features, train_labels)

#predict using test features set
ridge_pred = lasso.predict(test_features)

#calculate mae using ridge regression
mae = np.mean(np.absolute(np.subtract(ridge_pred, test_labels)))
print('\nMean Absolute Error when lambda is 10^-.5 = ', mae)

#finding the top 3 features
coef = dict(zip(train_features, lasso.coef_))
coef = {x:y for x,y in coef.items() if y!=0}
top_feature = sorted(coef, key=coef.get, reverse=True)[:10]
print("\nTop 3 features: ", top_feature)
