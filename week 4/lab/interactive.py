import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

d = pd.read_csv("../quiz/diabetes.csv")
main = sm.ols(formula="chol ~ age+frame",data=d).fit()
print(main.summary())
inter = sm.ols(formula="chol ~ age*frame",data=d).fit()
print(inter.summary())
inter = sm.ols(formula="chol ~ gender*frame",data=d).fit()
print(inter.summary())
inter = sm.ols(formula="chol ~ height*weight",data=d).fit()
print(inter.summary())
inter = sm.ols(formula="chol ~ age+weight+age*weight",data=d).fit()
print(inter.summary())