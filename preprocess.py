import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

propertyDF = pd.read_csv("property-tax-report.csv", sep=";")
addressDF = pd.read_csv("property-addresses.csv", sep=";")
censusDF = pd.read_csv("CensusLocalAreaProfiles2016.csv", encoding="ISO-8859-1")
subset = propertyDF.head(200000)
addr_subset = addressDF.head(100000)


def mergePropTax2006_2011(prop2011subset):
    # Making two subsets, one for data from 2006 and one for data from 2011
    subset_2011 = prop2011subset.drop(prop2011subset.loc[~prop2011subset['REPORT_YEAR'].isin([2011])].index, inplace=False)
    subset_2006 = prop2011subset.drop(prop2011subset.loc[~prop2011subset['REPORT_YEAR'].isin([2006])].index, inplace=False)

    # only keeping the columns we care about comparing between the years
    cols_to_keep = ['PID', 'CURRENT_LAND_VALUE', 'CURRENT_IMPROVEMENT_VALUE', 'REPORT_YEAR']
    subset_2006 = subset_2006[cols_to_keep]

    # merging the two datasets on PID
    mergedPropertyDF = subset_2006.merge(subset_2011, on='PID', how='inner')

    # getting the delta values by subtracting the two datasets
    mergedPropertyDF['CURRENT_LAND_VALUE'] = mergedPropertyDF['CURRENT_LAND_VALUE_y'] - mergedPropertyDF['CURRENT_LAND_VALUE_x']
    mergedPropertyDF['CURRENT_IMPROVEMENT_VALUE'] = mergedPropertyDF['CURRENT_IMPROVEMENT_VALUE_y'] - mergedPropertyDF['CURRENT_IMPROVEMENT_VALUE_x']

    print(mergedPropertyDF.shape)


mergePropTax2006_2011(prop2011subset)


def mergePropTax(prop2011subset):
    # defining the columns we will be keeping in the 2006-2013 property tax dataset
    cols_to_keep = ['PID', 'CURRENT_LAND_VALUE', 'CURRENT_IMPROVEMENT_VALUE', 'PREVIOUS_LAND_VALUE','PREVIOUS_IMPROVEMENT_VALUE', 'REPORT_YEAR' ]
    prop2011subset = prop2011subset[cols_to_keep]

    # dropping all columns where the years are not 2011 and 2016, respectively, in both property tax datasets
    prop2011subset.drop(prop2011subset.loc[prop2011subset['REPORT_YEAR'] != 2011].index, inplace=True)
    subset.drop(subset.loc[subset['REPORT_YEAR'] != 2016].index, inplace=True)

    # merging the two datasets on PID
    mergedPropertyDF = prop2011subset.merge(subset, on='PID', how='inner')

    # getting the delta values by subtracting the two datasets (Note: the 2006-2013 property tax report dataset does not have any previous land value
    # or previous improvement value entries
    mergedPropertyDF['CURRENT_LAND_VALUE'] = mergedPropertyDF['CURRENT_LAND_VALUE_y'] - mergedPropertyDF['CURRENT_LAND_VALUE_x']
    mergedPropertyDF['CURRENT_IMPROVEMENT_VALUE'] = mergedPropertyDF['CURRENT_IMPROVEMENT_VALUE_y'] - mergedPropertyDF['CURRENT_IMPROVEMENT_VALUE_x']
    mergedPropertyDF['PREVIOUS_LAND_VALUE'] = mergedPropertyDF['PREVIOUS_LAND_VALUE_y'] - mergedPropertyDF['PREVIOUS_LAND_VALUE_x']
    mergedPropertyDF['PREVIOUS_IMPROVEMENT_VALUE'] = mergedPropertyDF['PREVIOUS_IMPROVEMENT_VALUE_y'] - mergedPropertyDF['PREVIOUS_IMPROVEMENT_VALUE_x']

    # mergedPropertyDF = mergedPropertyDF['PID', 'CURRENT_LAND_VALUE', 'CURRENT_IMPROVEMENT_VALUE', 'PREVIOUS_LAND_VALUE','PREVIOUS_IMPROVEMENT_VALUE', 'REPORT_YEAR']
    print(mergedPropertyDF.columns.values)

    print(mergedPropertyDF.head(10).to_string())

# mergePropTax(prop2011subset)


def addCensus(data):
    mandarinColumn = []
    avgIncomeColumn = []
    lowIncomeColumn = []
    bachelorsDegreeColumn = []
    totalLabourForceColumn = []
    fullTimeWorkersColumn = []
    for row in data.itertuples():
        # i have no idea why region is _2
        # region also has trailing whitespace for some reason
        if row._2 is not None:
            region = row._2 + ' '

            # Mother tongue for mandarin is row 730 or id = 712
            num_mandarin = censusDF.iloc[728][region]
            mandarinColumn.append(num_mandarin)

            # Average income is row 1883, id = 1858, but need to go back 2? Using row 1881.
            num_income = censusDF.iloc[1881][region]
            avgIncomeColumn.append(num_income)

            # in low income is row 2506
            low_income = censusDF.iloc[2504][region]
            lowIncomeColumn.append(low_income)

            # bachelors degree is row 4254
            bachelors_degree = censusDF.iloc[4252][region]
            bachelorsDegreeColumn.append(bachelors_degree)

            # total labour force over the age of 15 is row 2233
            totalLabour_force = censusDF.iloc[2231][region]
            totalLabourForceColumn.append(totalLabour_force)

            #total people that worked full time in 2015 is row 2204
            fullTimeWorkers = censusDF.iloc[2202][region]
            fullTimeWorkersColumn.append(fullTimeWorkers)


        else:
            mandarinColumn.append(0)
            avgIncomeColumn.append(0)
            lowIncomeColumn.append(0)
            bachelorsDegreeColumn.append(0)
            totalLabourForceColumn.append(0)
            fullTimeWorkersColumn.append(0)

    data['native-mandarin'] = mandarinColumn
    data['avg-income'] = avgIncomeColumn
    data['low-income'] = lowIncomeColumn
    data['bachelors-degree'] = bachelorsDegreeColumn
    data['total-labour-force'] = totalLabourForceColumn
    data['full-time-workers'] = fullTimeWorkersColumn

    return

# pass in the dataframe and the year that you want to filter by
def filterByYear(df, year):
    filterYear = year
    df.drop(df.loc[df['REPORT_YEAR'] != filterYear].index, inplace=True)
    df.drop(df.loc[df['Geo Local Area'] == 0].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return

def dropColumns(df):
    df.drop(['NARRATIVE_LEGAL_LINE1'], inplace=True, axis=1)
    df.drop(['NARRATIVE_LEGAL_LINE2'], inplace=True, axis=1)
    df.drop(['NARRATIVE_LEGAL_LINE3'], inplace=True, axis=1)
    df.drop(['NARRATIVE_LEGAL_LINE4'], inplace=True, axis=1)
    df.drop(['NARRATIVE_LEGAL_LINE5'], inplace=True, axis=1)
    df.drop(['BLOCK'], inplace=True, axis=1)
    df.drop(['DISTRICT_LOT'], inplace=True, axis=1)
    df.drop(['LOT'], inplace=True, axis=1)
    df.drop(['FROM_CIVIC_NUMBER'], inplace=True, axis=1)
    df.drop(['TO_CIVIC_NUMBER'], inplace=True, axis=1)
    df.drop(['NEIGHBOURHOOD_CODE'], inplace=True, axis=1)
    df.drop(['PID'], inplace=True, axis=1)
    df.drop(['FOLIO'], inplace=True, axis=1)
    df.drop(['ZONE_NAME'], inplace=True, axis=1)
    df.drop(['PLAN'], inplace=True, axis=1)
    df.drop(['CIVIC_NUMBER'], inplace=True, axis=1)
    df.drop(['Geom'], inplace=True, axis=1)
    df.drop(['P_PARCEL_ID'], inplace=True, axis=1)
    df.drop(['PCOORD'], inplace=True, axis=1)
    df.drop(['SITE_ID'], inplace=True, axis=1)
    df.drop(['LAND_COORDINATE'], inplace=True, axis=1)
    df.drop(['STD_STREET'], inplace=True, axis=1)


def hotEncode(old_data):
    data_subset = pd.DataFrame()
    counter = 0
    for i in old_data:
        if (old_data[i].dtype == 'object' or len(old_data[i].unique()) < 20):
            encode = pd.get_dummies(old_data[i])
            for j in range(len(encode.columns)):
                encoded_name = encode.columns.values[j]
                encoded_columns_name = str(i) + "_" + str(encoded_name)
                data_subset.insert(counter + j, column=encoded_columns_name, value=encode[encoded_name], allow_duplicates=True)
            counter += len(encode.columns)
        else:
            data_subset.insert(counter, column=i, value=old_data[i])
            counter += 1
    return data_subset

# addRegionColumn()
# print(addr_subset)
# changeCardinal()
# changeCivic()
# addr_subset.rename(columns={'STD_STREET':'STREET_NAME'}, inplace=True)
# print(addr_subset)

sqlcode = '''
select *
from addr_subset
inner join subset on subset.LAND_COORDINATE=addr_subset.PCOORD
'''

newdf = ps.sqldf(sqlcode, locals())
# print(newdf.to_string())

# print(pd.merge(propertyDF, addr_subset, on='STREET_NAME').to_string())

# print(subset.to_string())
# print(censusDF.to_string())

addCensus(newdf)
dropColumns(newdf)
filterByYear(newdf, 2015)
newdf = pd.get_dummies(newdf, columns = ['ZONE_CATEGORY', 'Geo Local Area', 'LEGAL_TYPE'],
                             prefix=['ZONE_CATEGORY', 'REGION', 'LEGAL_TYPE'])
newdf.dropna(axis=0, how='any', inplace=True)
print(newdf.columns)
print(newdf.shape)

newdf.to_csv('new_output.csv', index=False)