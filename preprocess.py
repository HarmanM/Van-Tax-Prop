from region import getRegion
import pandas as pd
import pandasql as ps
import math
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

propertyDF = pd.read_csv("property-tax-report.csv", sep=";")
addressDF = pd.read_csv("property-addresses.csv", sep=";")
subset = propertyDF.head(20000)
addr_subset = addressDF.head(5000)

def addRegionColumn():
    region_list = []

    for row in subset.itertuples():
        postalCode = str(row.PROPERTY_POSTAL_CODE)
        civicNumber = str(int(row.TO_CIVIC_NUMBER))
        streetNumber = str(row.STREET_NAME)
        print(civicNumber + " " + streetNumber)
        address = str(civicNumber) + " " + str(streetNumber) + " VANCOUVER BC"
        region_list.append(getRegion(address))
        print(address)
    subset.insert(0, "REGION", region_list, allow_duplicates=True)

def changeCardinal():
    count = 0
    for row in addr_subset.itertuples():
        addr = str(row.STD_STREET)
        if addr[0:2] == "N " or addr[0:2] == "E " or addr[0:2] == "S " or addr[0:2] == "W ":
            cardinal = addr[0]
            addr_subset.at[count, 'STD_STREET'] = ''.join(addr[2:] + " " + cardinal)
        count = count + 1
    return

def changeCivic():
    count = 0
    for row in subset.itertuples():
        civic_code_to = row.TO_CIVIC_NUMBER
        civic_code_from = row.FROM_CIVIC_NUMBER
        if (not isinstance(civic_code_to, str) and not isinstance(civic_code_from, str)):
            print("Civic To: ", civic_code_to)
            print("Civic From: ", civic_code_from)
            civic_code_to = float(civic_code_to)
            civic_code_from = float(civic_code_from)
            if (math.isnan(civic_code_to)):
                print("Went in one", civic_code_from)
                civic_code_to = civic_code_from
                subset.at[count, 'TO_CIVIC_NUMBER'] = civic_code_to
            elif (math.isnan(civic_code_from)):
                print("Went in two", civic_code_to)
                civic_code_from = civic_code_to
                subset.at[count, 'FROM_CIVIC_NUMBER'] = civic_code_from
            elif (int(civic_code_from) < int(civic_code_to)):
                tmp = civic_code_from
                civic_code_from = civic_code_to
                civic_code_to = tmp
                print("CIVIC CODE: ", civic_code_from, civic_code_to)
                subset.at[count, 'TO_CIVIC_NUMBER'] = civic_code_to
                subset.at[count, 'FROM_CIVIC_NUMBER'] = civic_code_from
        count = count + 1
    return


#addRegionColumn()
#print(addr_subset)
changeCardinal()
changeCivic()
addr_subset.rename(columns={'STD_STREET':'STREET_NAME'}, inplace=True)
#print(addr_subset)

sqlcode = '''
select *
from addr_subset
inner join subset on subset.STREET_NAME=addr_subset.STREET_NAME
where addr_subset.CIVIC_NUMBER <= subset.TO_CIVIC_NUMBER and addr_subset.CIVIC_NUMBER >= subset.FROM_CIVIC_NUMBER
'''

newdf = ps.sqldf(sqlcode, locals())
print(newdf)

#print(pd.merge(propertyDF, addr_subset, on='STREET_NAME').to_string())

#print(subset.to_string())