import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import statsmodels.api as sma
import statsmodels.sandbox.tools.cross_val as cross_val
import datetime

file = open('validation.txt','w')

print(datetime.datetime.now(), file=file)
print(datetime.datetime.now())
d = pd.read_csv("cleaned.csv")
d = d.reindex(np.random.permutation(d.index))
d['ReleaseMonth'] = d['ReleaseMonth'].astype('str')
d['ReleaseYear'] = d['ReleaseYear'].astype('str')

formula = "ReservesLevel ~ ReleaseYear*Platform*ReleaseMonth*Edition*RelativeWeek"
output = "ReservesLevel"

# model = sma.load('best_model.pickle')

def loocv(d, formula, output):
    print('processing loocv', file=file)
    print('processing loocv')
    loo = cross_val.LeaveOneOut(len(d.index))
    error_sum = 0
    for train_index, test_index in loo:
        # print ("TRAIN:", train_index, "TEST:", test_index)
        a_train, a_test = cross_val.split(train_index, test_index, d)
        d_train = pd.DataFrame(a_train, columns=d.columns)
        d_test = pd.DataFrame(a_test, columns=d.columns)
        for x in d.columns:
            d_train[x] = d_train[x].astype(d[x].dtypes.name)
            d_test[x] = d_test[x].astype(d[x].dtypes.name)
        nuc = sm.ols(formula, data=d_train).fit()
        y = nuc.predict(d_test)
        error_sum += (y[0] - d_test['ReservesLevel'][0]) ** 2
    print("loocv MSE= ", (error_sum / len(d.index)), file=file)
    print("loocv MSE= ", (error_sum / len(d.index)))


def kFold(d, formula, output, size):
    print('processing kFold: ' + str(size), file=file)
    print('processing kFold: ' + str(size))
    loo = cross_val.KFold(len(d.index), size)
    error_sum = 0
    for train_index, test_index in loo:
        # print ("TRAIN:", train_index, "TEST:", test_index)
        a_train, a_test = cross_val.split(train_index, test_index, d)
        d_train = pd.DataFrame(a_train, columns=d.columns)
        d_test = pd.DataFrame(a_test, columns=d.columns)
        for x in d.columns:
            d_train[x] = d_train[x].astype(d[x].dtypes.name)
            d_test[x] = d_test[x].astype(d[x].dtypes.name)
        nuc = sm.ols(formula, data=d_train).fit()
        y = nuc.predict(d_test)
        error_sum += ((y - d_test['ReservesLevel']) ** 2).sum() / len(d_test.index)
    print("k-Fold {} MSE= {}".format(size,(error_sum / size)), file=file)
    print("k-Fold {} MSE= {}".format(size, (error_sum / size)))

print("validating forward selected : {}".format(formula), file=file)
print("validating forward selected: {}".format(formula))
loocv(d, formula, output)
print(datetime.datetime.now(), file=file)
print(datetime.datetime.now())
kFold(d, formula, output, 5)
print(datetime.datetime.now(), file=file)
print(datetime.datetime.now())
kFold(d, formula, output, 10)
print(datetime.datetime.now(), file=file)
print(datetime.datetime.now())

formula = 'ReservesLevel ~ Channel*Edition*Platform*ReleaseMonth*RelativeWeek'
print("validating backward selected : {}".format(formula), file=file)
print("validating backward selected: {}".format(formula))
loocv(d, formula, output)
print(datetime.datetime.now(), file=file)
print(datetime.datetime.now())
kFold(d, formula, output, 5)
print(datetime.datetime.now(), file=file)
print(datetime.datetime.now())
kFold(d, formula, output, 10)
print(datetime.datetime.now(), file=file)
print(datetime.datetime.now())
file.flush()
file.close()