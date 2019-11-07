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
subset = propertyDF.head(250000)
addr_subset = addressDF.head(100000)


def changeCardinal():
    count = 0
    for row in addr_subset.itertuples():
        addr = str(row.STD_STREET)
        if addr[0:2] == "N " or addr[0:2] == "E " or addr[0:2] == "S " or addr[0:2] == "W ":
            cardinal = addr[0]
            addr_subset.at[count, 'STD_STREET'] = ''.join(addr[2:] + " " + cardinal)
        count = count + 1
    return


def changeDrive():
    count = 0
    for row in addr_subset.itertuples():
        addr = str(row.STD_STREET)
        if re.search("\s{1}(Drive)", addr):
            cardinal = addr[0]
            addr_subset.at[count, 'STD_STREET'] = ''.join(addr[2:] + " " + cardinal)
        count = count + 1
    return


def changeCivic():
    count = 0
    for row in subset.itertuples():
        civic_code_to = (row.TO_CIVIC_NUMBER)
        civic_code_from = (row.FROM_CIVIC_NUMBER)

        if str(civic_code_to).isnumeric() and str(civic_code_from).isnumeric():
            print("Civic To: ", civic_code_to)
            print("Civic From: ", civic_code_from)
            civic_code_to = float(civic_code_to)
            civic_code_from = float(civic_code_from)

            if (math.isnan(civic_code_to)):
                print("went in one", civic_code_from)
                civic_code_to = civic_code_from
                subset.at[count, 'TO_CIVIC_NUMBER'] = civic_code_to

            elif (math.isnan(civic_code_from)):
                print("Went in two", civic_code_to)
                civic_code_from = civic_code_to
                subset.at[count, 'FROM_CIVIC_NUMBER'] = civic_code_from

            elif civic_code_from < civic_code_to:
                print("went in three")
                tmp = civic_code_from
                civic_code_from = civic_code_to
                civic_code_to = tmp
                print("CIVIC CODE: ", civic_code_from, civic_code_to)
                subset.at[count, 'TO_CIVIC_NUMBER'] = civic_code_to
                subset.at[count, 'FROM_CIVIC_NUMBER'] = civic_code_from
        count = count + 1
    return


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
