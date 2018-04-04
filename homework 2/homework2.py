import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as em

df = pd.read_csv("data.csv",index_col=False)
pd.to_datetime(df['YEARMODA'],format="%Y%m%d")

#cleaning up null values in data
df = df[df["TEMP"] != 9999.9]
df = df[df["DEWP"] != 9999.9]
df = df[df["SLP"] != 9999.9]
df = df[df["MAX"] != 9999.9]
df = df[df["MIN"] != 9999.9]
df = df[df["MXSPD"] != 999.9]


#get sample set
smpl = df.sample(100)

#sample removed from original
df2 = df.drop(smpl.index)

#
res = sm.ols(formula = "TEMP ~ DEWP",data=df2).fit()
print(res.summary())

y2 = res.predict(smpl['DEWP'])

print("rmse:",em.rmse(smpl['DEWP'], y2))


plt.scatter(df2['TEMP'],df2['DEWP'],label="data")
plt.scatter(smpl["TEMP"],smpl["DEWP"], color="purple",label="samples")
plt.scatter(y2,smpl["DEWP"], color="red",label = "predicted")
plt.plot(res.fittedvalues, df2['DEWP'],label="fitted", color="black")
plt.legend()
plt.xlabel('DEWP')
plt.ylabel('TEMP')
plt.savefig('partA.png')
plt.show()

plt.clf()
plt.boxplot(df['TEMP'])
plt.savefig('temp.png')
plt.show()

plt.clf()
plt.boxplot(df['DEWP'])
plt.savefig('dewp.png')
plt.show()

plt.clf()
plt.boxplot(df['SLP'])
plt.savefig('slp.png')
plt.show()

plt.clf()
plt.boxplot(df['MIN'])
plt.savefig('min.png')
plt.show()

plt.clf()
plt.boxplot(df['MAX'])
plt.savefig('max.png')
plt.show()

plt.clf()
plt.scatter(df2['TEMP'],df2['SLP'],label="temp_slp")
plt.xlabel('TEMP')
plt.ylabel('SLP')
plt.savefig('temp_slp.png')
plt.show()

plt.clf()
plt.scatter(df2['TEMP'],df2['MXSPD'],label="temp_mxspd")
plt.xlabel('TEMP')
plt.ylabel('MXSPD')
plt.savefig('temp_mxspd.png')
plt.show()

plt.clf()
plt.scatter(df2['DEWP'],df2['MXSPD'],label="dewp_mxspd")
plt.xlabel('DEWP')
plt.ylabel('MXSPD')
plt.savefig('dewp_mxspd.png')
plt.show()

plt.clf()
plt.scatter(df2['DEWP'],df2['SLP'],label="dewp_slp")
plt.xlabel('DEWP')
plt.ylabel('SLP')
plt.savefig('dewp_slp.png')
plt.show()
