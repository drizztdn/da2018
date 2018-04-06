import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma
import seaborn as sns; 

pd.options.mode.chained_assignment = None

def fx(x):
    if x:
        return 'Down'
    else:
        return 'Up'

df=pd.read_csv("smarket.csv")
trn = df.copy()

print(df.describe())


sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df)
g.savefig('pairwise.png')

print(df.corr())

res = sm.glm(formula="Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5",data=df,family=sma.families.Binomial()).fit()
print(res.summary())

ypnew = res.predict(df)
df['pred'] = ypnew
df['pred_dir'] = ''
df['pred_correct'] = False
df['pred_dir'][df['pred'] > .5] = "Down"
df['pred_dir'][df['pred'] <= .5] = "Up"
df['pred_correct'] = df['pred_dir'] == df['Direction']
#print(df)

print(df.groupby('pred_correct')['pred'].count())

smpl = trn[trn["Year"] == 2005]
trn = trn.drop(smpl.index)

res2 = sm.glm(formula="Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5",data=trn,family=sma.families.Binomial()).fit()
yp2new = res2.predict(smpl)

smpl['pred'] = yp2new
smpl['pred_dir'] = ''
smpl['pred_correct'] = False
smpl['pred_dir'][smpl['pred'] > .5] = "Down"
smpl['pred_dir'][smpl['pred'] <= .5] = "Up"
smpl['pred_correct'] = smpl['pred_dir'] == smpl['Direction']

print(smpl.groupby('pred_correct')['pred'].count())

#print(yp2new)