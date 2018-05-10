import statsmodels.api as sma
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("fixed.csv")
df['ReleaseMonth'] = df['ReleaseMonth'].astype('str')
df['ReleaseYear'] = df['ReleaseYear'].astype('str')
print(df.dtypes)

cas = df[df['GameType'] == 'Casual']
noncas = df[df['GameType'] == 'NonCasual']


plt.scatter(cas['RelativeWeek'],cas['ReservesLevel'],label="data")
plt.ylabel("Reserves")
plt.xlabel("Relative Week")
plt.title("Casual Games")
plt.show()
plt.clf()

plt.scatter(noncas['RelativeWeek'],noncas['ReservesLevel'],label="data")
plt.ylabel("Reserves")
plt.xlabel("Relative Week")
plt.title("Non-Casual Games")
plt.show()