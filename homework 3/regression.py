import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma

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
    print(smpl.groupby('pred_correct')['pred'].count())
    return smpl