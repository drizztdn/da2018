import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
###########
c1=pd.read_csv("chickwts.csv")
w1 = c1["weight"]
print(w1.mean())
print(w1.median())
f1 = c1["feed"]
print(f1.value_counts())
print(f1.value_counts()/f1.size)
print(f1.value_counts()/f1.size*100)
print(w1.quantile(0.8))
print(w1.describe())