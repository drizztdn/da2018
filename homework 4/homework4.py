import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma


def test_interactions(f, outcome, predictors):
    d = pd.read_csv(f)
    f = outcome +" ~ " + "+".join(predictors)
    f2 = outcome +" ~ " + "*".join(predictors)

    no_inter = sm.ols(formula=f, data=d).fit()
    inter = sm.ols(formula=f2, data=d).fit()

    print("no interactions")
    print(no_inter.summary())

    print("interactions")
    print(inter.summary())
    return no_inter, inter

print('a')
no_interactions, interactions  = test_interactions("cats.csv",'Hwt',['Bwt', 'Sex'])


tc = pd.DataFrame(columns=['Bwt','Sex'])
tc.loc[0] = [3.4, 'F']

outcome = interactions.predict(tc)

print(outcome)

print('b')
no_interactions, interactions  = test_interactions("trees.csv",'Volume',['Girth', 'Height'])

no_interactions, interactions  = test_interactions("trees.csv",'Volume',['np.log(Girth)', 'np.log(Height)'])