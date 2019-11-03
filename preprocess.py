from region import getRegion
import pandas as pd
import pandasql as ps
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

propertyDF = pd.read_csv("property-tax-report.csv", sep=";")
addressDF = pd.read_csv("property-addresses.csv", sep=";")
subset = propertyDF.head(2500)
addr_subset = addressDF.head(2500)

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

#addRegionColumn()
#print(addr_subset)
changeCardinal()
addr_subset.rename(columns={'STD_STREET':'STREET_NAME'}, inplace=True)
#print(addr_subset)

sqlcode = '''
select *
from addr_subset
inner join subset on subset.STREET_NAME=addr_subset.STREET_NAME
where addr_subset.CIVIC_NUMBER <= subset.TO_CIVIC_NUMBER and addr_subset.CIVIC_NUMBER >= subset.FROM_CIVIC_NUMBER
group by addr_subset.CIVIC_NUMBER
'''

newdf = ps.sqldf(sqlcode, locals())
print(newdf.to_string())

#print(pd.merge(propertyDF, addr_subset, on='STREET_NAME').to_string())
