import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as em

df = pd.read_csv("data.csv",index_col=False)
pd.to_datetime(df['YEARMODA'],format="%Y%m%d")

#cleaning up null values in data
df = df[df["DEWP"] != 9999.9]
df = df[df["SLP"] != 9999.9]

print(df)

smpl = df.sample(100)

print(smpl)

df2 = df.drop(smpl.index)

print(df2)

res = sm.ols(formula = "TEMP ~ DEWP",data=df2).fit()
print(res.summary())

yp2 = res.predict(smpl['DEWP'])

plt.scatter(df2['TEMP'],df2['DEWP'],label="data")
plt.scatter(smpl["TEMP"],smpl["DEWP"], color="purple",label="samples")
plt.scatter(yp2,smpl["DEWP"], color="red",label = "predicted")
plt.legend()
plt.xlabel('DEWP')
plt.ylabel('TEMP')
plt.show()