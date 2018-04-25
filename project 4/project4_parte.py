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
    print('processing loocv')
    loo = cross_val.LeaveOneOut(len(d.index))
    error_sum = 0
    for train_index, test_index in loo:
        # print ("TRAIN:", train_index, "TEST:", test_index)
        a_train, a_test = cross_val.split(train_index, test_index, d)
        d_train = pd.DataFrame(a_train, columns=d.columns)
        d_test = pd.DataFrame(a_test, columns=d.columns)
        d_train['ReleaseMonth'] = d_test['ReleaseMonth'].astype('str')
        d_train['Region'] = d_test['Region'].astype('str')
        d_train['Platform'] = d_test['Platform'].astype('str')
        d_train['Channel'] = d_test['Channel'].astype('str')
        d_train['Edition'] = d_test['Edition'].astype('str')
        d_train['RelativeWeek'] = d_test['RelativeWeek'].astype('int')
        d_test['ReleaseMonth'] = d_test['ReleaseMonth'].astype('str')
        d_test['Region'] = d_test['Region'].astype('str')
        d_test['Platform'] = d_test['Platform'].astype('str')
        d_test['Channel'] = d_test['Channel'].astype('str')
        d_test['Edition'] = d_test['Edition'].astype('str')
        d_test['RelativeWeek'] = d_test['RelativeWeek'].astype('int')
        nuc = sm.ols(formula, data=d_train).fit()
        y = nuc.predict(d_test)
        error_sum += (y[0] - d_test['ReservesLevel'][0]) ** 2
    print("MSE= ", (error_sum / len(d.index)))


def kFold(d, formula, output, size):
    print('processing kFold: ' + str(size))
    loo = cross_val.KFold(len(d.index), size)
    error_sum = 0
    for train_index, test_index in loo:
        # print ("TRAIN:", train_index, "TEST:", test_index)
        a_train, a_test = cross_val.split(train_index, test_index, d)
        d_train = pd.DataFrame(a_train, columns=d.columns)
        d_test = pd.DataFrame(a_test, columns=d.columns)
        # d_train['ReleaseMonth'] = d_test['ReleaseMonth'].astype('str')
        # d_train['Region'] = d_test['Region'].astype('str')
        # d_train['Platform'] = d_test['Platform'].astype('str')
        # d_train['Channel'] = d_test['Channel'].astype('str')
        # d_train['Edition'] = d_test['Edition'].astype('str')
        # d_train['RelativeWeek'] = d_test['RelativeWeek'].astype('int')
        # d_test['ReleaseMonth'] = d_test['ReleaseMonth'].astype('str')
        # d_test['Region'] = d_test['Region'].astype('str')
        # d_test['Platform'] = d_test['Platform'].astype('str')
        # d_test['Channel'] = d_test['Channel'].astype('str')
        # d_test['Edition'] = d_test['Edition'].astype('str')
        # d_test['RelativeWeek'] = d_test['RelativeWeek'].astype('int')
        for x in d.columns:
            d_train[x] = d_train[x].astype(d[x].dtypes.name)
            d_test[x] = d_test[x].astype(d[x].dtypes.name)
        nuc = sm.ols('ReservesLevel ~ Platform*Region*ReleaseMonth*Channel*Edition*RelativeWeek', data=d_train).fit()
        y = nuc.predict(d_test)
        error_sum += ((y - d_test['ReservesLevel']) ** 2).sum() / len(d_test.index)
    print("MSE= ", (error_sum / size))

# loocv(d, formula, output)
kFold(d, formula, output, 5)
# kFold(d, formula, output, 10)