import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as em

def fx(x, y):
    return em.rmse(x, y)

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

# run linear regression
res = sm.ols(formula = "TEMP ~ DEWP",data=df2).fit()
print(res.summary())

# predict values based on extracted sample set
y2 = res.predict(smpl['DEWP'])
y2.to_csv('predicted.csv')
y2.columns = ['pred_temp']

# add prediction to sample set for comparison
smpl['pred_temp'] = y2

# remove columns not needed for this sample
del smpl['SLP']
del smpl['STP']
del smpl['MAX']
del smpl['MIN']
del smpl['GUST']
del smpl['FRSHTT']
del smpl['VISIB']
del smpl['WDSP']
del smpl['MXSPD']
del smpl['PRCP']
del smpl['SNDP']
smpl.to_csv('sample.csv')

print("rmse:",em.rmse(smpl['TEMP'], smpl['pred_temp']))

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
