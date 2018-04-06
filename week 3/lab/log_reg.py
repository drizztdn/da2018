import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm 
import statsmodels.api as sma

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    axes.set_autoscale_on(False)
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    
d=pd.read_csv("default.csv")
# Add a new column DefaultYes which is 1 for Yes and 0 for No
d['DefaultYes'] = d['default'].map({'Yes': 1, 'No': 0})

# As a linear regression
print("dataframe", d)
res = sm.ols(formula="DefaultYes ~ balance",data=d).fit()
print(res.summary())
print(res.params.values)

# Plot the data points
plt.scatter(d["balance"],d["DefaultYes"])
# Plot a line using the coefficients as slope and intercept
abline(res.params.values[1],res.params.values[0])
plt.show()

# Logistic fit
res2 = sm.glm(formula="default ~ balance",data=d,family=sma.families.Binomial()).fit()
print(res2.summary())
# Build a new dataframe with balances from 0 to 3000 to predict and draw
x1new = pd.DataFrame(np.hstack((np.arange(0,3000))))
x1new.columns=["balance"]
yp2new = res2.predict(x1new)
# Note that ['default[No]', 'default[Yes]'] 
plt.scatter(d["balance"],d["DefaultYes"])
plt.plot(x1new,1-yp2new)
plt.show()

plt.clf()
#multilogistic fit
res3 = sm.glm(formula="default ~ balance+student",data=d,family=sma.families.Binomial()).fit()
print(res3.summary())
x3new = pd.DataFrame(np.hstack((np.arange(0,2500,10).reshape(250,1),np.repeat("Yes",250).reshape(250,1))))
x3new.columns=["balance","student"]
x3new[["balance"]] = x3new[["balance"]].astype(float)
x3new[["student"]] = x3new[["student"]].astype(str)
yp3new = res3.predict(x3new)
plt.plot(x3new["balance"],1-yp3new, color="red")
x4new = pd.DataFrame(np.hstack((np.arange(0,2500,10).reshape(250,1),np.repeat("No",250).reshape(250,1))))
x4new.columns=["balance","student"]
x4new[["balance"]] = x4new[["balance"]].astype(float)
x4new[["student"]] = x4new[["student"]].astype(str)
yp4new = res3.predict(x4new)
plt.plot(x4new["balance"],1-yp4new, color="blue")
plt.show()