import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma

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

chol1 = sm.ols(formula="chol ~ 1",data=d).fit()
chol2 = sm.ols(formula="chol ~ age",data=d).fit()
chol3 = sm.ols(formula="chol ~ age+frame",data=d).fit()
chol4 = sm.ols(formula="chol ~ age*frame",data=d).fit()
print(sma.stats.anova_lm(chol1,chol2,chol3,chol4))