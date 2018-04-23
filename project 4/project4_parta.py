import pandas as pd
import statsmodels.formula.api as sm

def clean(df):
    df['Region'] = df['Region'].str.strip()
    df['Installment'] = df['Installment'].str.strip()
    df['Edition'] = df['Edition'].str.strip()
    df['Platform'] = df['Platform'].str.strip()
    return df

def test_interactions(d, outcome, predictors,col_dt = None):
    if col_dt != None:
        for i in col_dt:
            d[i[0]] = d[i[0]].astype(i[1])
    f = outcome +" ~ " + "+".join(predictors)
    f2 = outcome +" ~ " + "*".join(predictors)

    no_inter = sm.ols(formula=f, data=d).fit()
    inter = sm.ols(formula=f2, data=d).fit()

    print("no interactions: {}".format(f))
    print(no_inter.summary())

    print("interactions: {}".format(f2))
    print(inter.summary())
    return no_inter, inter


df = clean(pd.read_csv("DA_Clean NCSA Reserves_4.14.18-FINAL.csv"))
# print(df['Edition'].unique())
# print(df['Installment'].unique())
# print(df['Channel'].unique())
# print(df['Platform'].unique())
# print(df['ReleaseYear'].unique())
# print(df['ReleaseMonth'].unique())
# print(df['GameType'].unique())
# print(df['Region'].unique())

no_inter, inter = test_interactions(df, 'ReservesLevel', ['Edition', 'Channel', 'Platform', 'GameType',
                        'ReleaseMonth'], [['ReleaseMonth', 'str'], ['ReleaseYear', 'str']])

f = open('interactions.txt','w')
print(inter.summary(), file=f)