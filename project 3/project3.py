import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma
import seaborn as sns; 
pd.options.mode.chained_assignment = None


df=pd.read_csv("SampleData.csv")

print(df.dtypes)