import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
d=pd.read_csv("mtcars.csv")
#print("dataframe", d)
res1 = sm.ols(formula="mpg ~ disp",data=d).fit()
print(res1.summary())
res2 = sm.ols(formula="mpg ~ disp+I(disp*disp)",data=d).fit()
print(res2.summary())
res3 = sm.ols(formula="mpg ~ disp+I(disp*disp)+ I(disp*disp*disp)",data=d).fit()
print(res3.summary())
res4 = sm.ols(formula="mpg ~ disp+I(disp*disp)+ I(disp*disp*disp)+ I(disp*disp*disp*disp)",data=d).fit()
print(res4.summary())
res5 = sm.ols(formula="mpg ~ np.log(hp)+am",data=d).fit()
print(res5.summary())


plt.scatter(d['mpg'],d['disp'],label="data")
plt.plot(res1.fittedvalues, d['disp'],label="linear", color="black")
plt.plot(res2.fittedvalues, d['disp'],label="quad",linestyle = '--', color="black")
plt.legend()
plt.xlabel('disp')
plt.ylabel('mpg')
plt.show()