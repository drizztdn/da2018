import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as em

y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])
x1 = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])
id = np.ones((8,1))
x = np.hstack((id,x1))
beta=(np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y))
print(beta)
yp1 = beta[0]+beta[1]*x1
print(np.hstack((x1,y,yp1)))

d = pd.DataFrame(np.hstack((x1,y)))
d.columns = ["x1","y"]
print("dataframe", d)
res = sm.ols(formula="y ~ x1",data=d).fit()
print("Summary", res.summary())
yp2 = res.predict(x1).values.reshape((8,1))
print(np.hstack((x1,y,yp1,yp2)))

plt.scatter(x1,y)
plt.plot(x1,yp2, color="blue")
#plt.show()

print()

y = np.array([[1.55],[0.42],[1.29],[0.73],[0.76],[-1.09],[1.41],[-0.32]])
x1 = np.array([[1.13],[-0.73],[0.12],[0.52],[-0.54],[-1.15],[0.20],[-1.09]])
x2 = np.array([[1],[0],[1],[1],[0],[1],[0],[1]])
#Manual
id = np.ones((8,1))
x = np.hstack((id,x1,x2))
print(x)
beta=(np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y))
print(beta)
yp1 = beta[0]+beta[1]*x1+beta[2]*x2
print(np.hstack((x,y,yp1)))
# Statsmodels
d = pd.DataFrame(np.hstack((x1,x2,y)))
d.columns = ["x1","x2","y"]
print("dataframe", d)
res = sm.ols(formula="y ~ x1+x2",data=d).fit()
yp2 = res.predict(np.hstack((x1,x2))).values.reshape((8,1))
print(np.hstack((x1,x2,y,yp1,yp2)))

print()
print("survey")
d=pd.read_csv("survey.csv")
d=d.rename(index=str,columns={"Wr.Hnd":"WrHnd"})
print("dataframe", d)
res = sm.ols(formula="Height ~ WrHnd+Sex+Smoke",data=d).fit()
print(res.summary())

print()
print("clocks")
d=pd.read_csv("clock.csv")
print("dataframe", d)
res = sm.ols(formula="Price ~ Bidders+Age",data=d).fit()
print(res.summary())