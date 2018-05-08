import statsmodels.api as sma
import pandas as pd

df = pd.read_csv("../project 4/cleaned.csv")
model = sma.load('best_model.pickle')


plt.scatter(df2['TEMP'],df2['DEWP'],label="data")
plt.scatter(smpl["TEMP"],smpl["DEWP"], color="purple",label="samples")
plt.scatter(y2,smpl["DEWP"], color="red",label = "predicted")
# plt.plot(model.fittedvalues, df2['DEWP'],label="fitted", color="black")
plt.legend()
plt.xlabel('weeks')
plt.ylabel('reserves')
plt.savefig('partA.png')
plt.show()