import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma

d = pd.read_csv("../quiz/diabetes.csv")

d = d.dropna(subset=['age'])
d = d.dropna(subset=['gender'])
d = d.dropna(subset=['height'])
d = d.dropna(subset=['weight'])
d = d.dropna(subset=['frame'])
d = d.dropna(subset=['waist'])
d = d.dropna(subset=['hip'])
d = d.dropna(subset=['location'])

dia = []
dia.append(sm.ols(formula="chol ~ age+gender+height",data=d).fit())
print(dia[0].summary())
dia.append(sm.ols(formula="chol ~ age+gender+height+weight",data=d).fit())
dia.append(sm.ols(formula="chol ~ age+gender+height+frame",data=d).fit())
dia.append(sm.ols(formula="chol ~ age+gender+height+waist",data=d).fit())
dia.append(sm.ols(formula="chol ~ age+gender+height+hip",data=d).fit())
dia.append(sm.ols(formula="chol ~ age+gender+height+location",data=d).fit())
for i in np.arange(1,6):
    print(sma.stats.anova_lm(dia[0],dia[i])['Pr(>F)'])