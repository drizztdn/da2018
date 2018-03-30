import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

chick = pd.read_csv("chickwts.csv")
rock = pd.read_csv("rock.csv")
stack = pd.read_csv("stackloss.csv")
survey = pd.read_csv("survey.csv")
titanic = pd.read_csv("titanic.csv")
trees = pd.read_csv("trees.csv")

tv = trees["Volume"]
print(tv.mean())
th = trees["Height"]
print(th.mean())

ta = titanic["Age"]
print(ta.std())

tg = trees["Girth"]
tgm = tg.median()
tgf = tg.quantile(.25)
print(tgm-tgf)

r1 = rock[["area","shape"]]
print("Correlation: ",r1.corr())