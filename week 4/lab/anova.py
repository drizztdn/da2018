import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma

d = pd.read_csv("../quiz/nuclear.csv")
d=d.rename(index=str,columns={"cum.n":"cumn"})
dia = []
dia.append(sm.ols("cost~date+cap+pt+ne", data=d).fit())
print(dia[0].summary())
dia.append(sm.ols("cost~date+cap+pt+ne+t1", data=d).fit())
dia.append(sm.ols("cost~date+cap+pt+ne+t2", data=d).fit())
dia.append(sm.ols("cost~date+cap+pt+ne+pr", data=d).fit())
dia.append(sm.ols("cost~date+cap+pt+ne+ct", data=d).fit())
dia.append(sm.ols("cost~date+cap+pt+ne+bw", data=d).fit())
dia.append(sm.ols("cost~date+cap+pt+ne+cumn", data=d).fit())
for i in np.arange(1,7):
    print(sma.stats.anova_lm(dia[0],dia[i]))