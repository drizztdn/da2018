import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma
import seaborn as sns; 

df=pd.read_csv("smarket.csv")

print(df.describe())


sns.set(style="ticks", color_codes=True)
#g = sns.pairplot(df)
#plt.show()

print(df.corr())

df['DirectionUp'] = df['Direction'].map({'Up': 1, 'Down': 0})

res = sm.glm(formula="DirectionUp ~ Lag1+Lag2+Lag3+Lag4+Lag5",data=df,family=sma.families.Binomial()).fit()
print(res.summary())

ypnew = res.predict(df)
ypnew['']

smpl = df[df["Year"] == 2005]
trn = df.drop(smpl.index)

res2 = sm.glm(formula="DirectionUp ~ Lag1+Lag2+Lag3+Lag4+Lag5",data=trn,family=sma.families.Binomial()).fit()
yp2new = res2.predict(smpl)

#print(yp2new)