import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma


cats=pd.read_csv("cats.csv")
diabetes=pd.read_csv("diabetes.csv")

cat_res = sm.ols(formula="Bwt ~ Hwt + I(Hwt*Hwt)", data=cats).fit()

print(cat_res.summary())

diabetes_res = sm.glm(formula="location ~ height + weight + chol",data=diabetes,family=sma.families.Binomial()).fit()

print(diabetes_res.summary())