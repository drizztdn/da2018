import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.api as sma
import statsmodels.sandbox.tools.cross_val as cross_val

d = pd.read_csv("DA_Clean NCSA Reserves_4.14.18-FINAL.csv")
d['ReleaseMonth'] = d['ReleaseMonth'].astype('str')
d['ReleaseYear'] = d['ReleaseYear'].astype('str')

formula = "ReservesLevel ~ Platform*Region*ReleaseMonth*Channel*Edition*RelativeWeek"
output = "ReservesLevel"

# model = sma.load('best_model.pickle')

def loocv(d, formula, output):
    loo = cross_val.LeaveOneOut(len(d.index))
    error_sum = 0

    for train_index, test_index in loo:
        print("TRAIN:", train_index, "TEST:", test_index)
        a_train, a_test = cross_val.split(train_index,test_index,d)
        d_train = pd.DataFrame(a_train,columns=d.columns)
        d_test = pd.DataFrame(a_test,columns=d.columns)
        for x in d.columns:
            d_train[x] = d_train[x].astype(d[x].dtypes.name)
            d_test[x] = d_test[x].astype(d[x].dtypes.name)
        nuc = sm.ols(formula, data=d_train).fit()
        y = nuc.predict(d_test)
        error_sum+= (y[0] - d_test[output][0])**2
    print("LOOCV MSE= ", (error_sum/len(d.index)))


def kFold(d, formula, output, size):
    loo = cross_val.KFold(len(d.index),size)
    error_sum = 0
    for train_index, test_index in loo:
        print ("TRAIN:", train_index, "TEST:", test_index)
        a_train, a_test = cross_val.split(train_index,test_index,d)
        d_train = pd.DataFrame(a_train,columns=d.columns)
        d_test = pd.DataFrame(a_test,columns=d.columns)
        for x in d.columns:
            d_train[x] = d_train[x].astype(d[x].dtypes.name)
            d_test[x] = d_test[x].astype(d[x].dtypes.name)
        nuc = sm.ols(formula, data=d_train).fit()
        y = nuc.predict(d_test)
        error_sum+= ((y - d_test[output])**2).sum()/len(d_test.index)
    print("k-Fold " + size +" MSE= ", (error_sum/size))

loocv(d, formula, output)
kFold(d, formula, output, 5)
kFold(d, formula, output, 10)