import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma
import seaborn as sns; 
pd.options.mode.chained_assignment = None

def train(trn):
    res = sm.glm(formula="Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5",data=trn,family=sma.families.Binomial()).fit()
    print(res.summary())
    return res

def predict(trn, smpl):
    ypnew = trn.predict(smpl)
    smpl['pred'] = ypnew
    smpl['pred_dir'] = ''
    smpl['pred_correct'] = False
    smpl['pred_dir'][smpl['pred'] > .5] = "Down"
    smpl['pred_dir'][smpl['pred'] <= .5] = "Up"
    smpl['pred_correct'] = smpl['pred_dir'] == smpl['Direction']
    print("total pred correct: ",smpl.groupby('pred_correct')['pred'].count())
    print("up correct: ",smpl['pred_correct'][smpl['Direction'] == 'Up'].describe())
    print("down correct:", smpl['pred_correct'][smpl['Direction'] == 'Down'].describe())
    return smpl

df=pd.read_csv("smarket.csv")
trn = df.copy()

print(df.describe(include='all'))
print(df.corr())

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df)
#g.savefig('pairwise.png')
#plt.show()

rgs = train(df)

df = predict(rgs, df)

smpl = trn[trn["Year"] == 2005]
trn = trn.drop(smpl.index)

rgs2 = train(trn)
smpl = predict(rgs2, smpl)
