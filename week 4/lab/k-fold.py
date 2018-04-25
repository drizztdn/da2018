import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.api as sma
import statsmodels.sandbox.tools.cross_val as cross_val

d=pd.read_csv("auto.csv")
loo = cross_val.KFold(len(d.index),20)
error_sum = 0
for train_index, test_index in loo:
    # print ("TRAIN:", train_index, "TEST:", test_index)
    a_train, a_test = cross_val.split(train_index,test_index,d)
    d_train = pd.DataFrame(a_train,columns=d.columns)
    d_test = pd.DataFrame(a_test,columns=d.columns)
    for x in d.columns:
        d_train[x] = d_train[x].astype(d[x].dtypes.name)
        d_test[x] = d_test[x].astype(d[x].dtypes.name)
    nuc = sm.ols("mpg~horsepower", data=d_train).fit()
    y = nuc.predict(d_test)
    error_sum+= ((y - d_test["mpg"])**2).sum()/len(d_test.index)

print( "MSE= ", (error_sum/20))