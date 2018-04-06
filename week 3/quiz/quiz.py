import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as em

boston = pd.read_csv("boston.csv")
rabbit = pd.read_csv("rabbit.csv")

res = sm.ols(formula = "medv ~ age",data=boston).fit()
print(res.summary())

rres = sm.ols(formula = "BPchange ~ Run",data=rabbit).fit()
print(rres.summary())