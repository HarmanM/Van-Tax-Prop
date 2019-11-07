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

def addCensus(data):
    mandarinColumn = []
    avgIncomeColumn = []
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
        else:
            mandarinColumn.append(0)
            avgIncomeColumn.append(0)
    data['native-mandarin'] = mandarinColumn
    data['avg-income'] = avgIncomeColumn
    return

# pass in the dataframe and the year that you want to filter by
def filterByYear(df, year):
    filterYear = year
    df.drop(df.loc[df['REPORT_YEAR'] != filterYear].index, inplace=True)
    df.drop(df.loc[df['Geo Local Area'] == 0].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return


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
filterByYear(newdf, 2015)

print(newdf.to_string())
print(newdf.shape)
