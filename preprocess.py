from region import getRegion
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

propertyDF = pd.read_csv("property-tax-report.csv", sep=";")
subset = propertyDF.head(25)

def addRegionColumn():
    region_list = []

    for row in subset.itertuples():
        postalCode = str(row.PROPERTY_POSTAL_CODE)
        civicNumber = str(int(row.TO_CIVIC_NUMBER))
        streetNumber = str(row.STREET_NAME)

        address = str(postalCode) + " " + str(civicNumber) + " " + str(streetNumber)
        region_list.append(getRegion(address))
        print(address)
    subset.insert(0, "REGION", region_list, allow_duplicates=True)

addRegionColumn()
print(subset)

