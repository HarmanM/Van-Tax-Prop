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
subset = propertyDF.head(25000)
addr_subset = addressDF.head(10000)

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
filterByYear(newdf, 2019)

print(newdf)
print(newdf.shape)
