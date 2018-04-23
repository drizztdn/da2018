import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

df = pd.read_csv("DA_Clean NCSA Reserves_4.14.18-FINAL.csv")

trans1 = sm.ols(formula='ReservesLevel ~ np.log(ReleaseMonth)', data=df).fit()
trans2 = sm.ols(formula='ReservesLevel ~ np.log(ReleaseYear)', data=df).fit()
trans3 = sm.ols(formula='ReservesLevel ~ I(ReleaseMonth*ReleaseMonth)', data=df).fit()
trans4 = sm.ols(formula='ReservesLevel ~ I(ReleaseMonth*ReleaseMonth*ReleaseMonth) + I(ReleaseMonth*ReleaseMonth)', data=df).fit()

print('ReservesLevel ~ np.log(ReleaseMonth)')
print(trans1.summary())
print('ReservesLevel ~ np.log(ReleaseYear)')
print(trans2.summary())
print('ReservesLevel ~ I(ReleaseMonth*ReleaseMonth)')
print(trans3.summary())
print('ReservesLevel ~ I(ReleaseMonth*ReleaseMonth*ReleaseMonth) + I(ReleaseMonth*ReleaseMonth)')
print(trans4.summary())