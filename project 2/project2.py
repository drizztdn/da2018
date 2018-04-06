import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as em

def fx(x, y):
    return em.rmse(x, y)

df = pd.read_csv("../homework 2/data.csv",index_col=False)

#setting types
#date column to a date
df['YEARMODA'] = df['YEARMODA'].astype(str)
df['YEARMODA'] = pd.to_datetime(df['YEARMODA'],format="%Y%m%d")
#string columns to strings
df['FRSHTT'] = df['FRSHTT'].astype(str)
df['PRCP'] = df['PRCP'].astype(str)

#cleaning up null values in data
df['TEMP'] = df['TEMP'].replace({9999.9: np.nan})
df['DEWP'] = df['DEWP'].replace({9999.9: np.nan})
df['SLP'] = df['SLP'].replace({9999.9: np.nan})
df['STP'] = df['STP'].replace({9999.9: np.nan})
df['VISIB'] = df['VISIB'].replace({999.9: np.nan})
df['WDSP'] = df['WDSP'].replace({999.9: np.nan})
df['MXSPD'] = df['MXSPD'].replace({999.9: np.nan})
df['GUST'] = df['GUST'].replace({999.9: np.nan})
df['MAX'] = df['MAX'].replace({9999.9: np.nan})
df['MIN'] = df['MIN'].replace({9999.9: np.nan})
df['PRCP'] = df['PRCP'].replace({99.9: np.nan})
df['SNDP'] = df['SNDP'].replace({999.9: np.nan})

# run linear regression
print(df.dtypes)
