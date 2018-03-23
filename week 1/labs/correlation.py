import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
######
o = pd.read_csv("orange.csv")
o1 = o[["age","circumference"]]
print(o1)
print("Correlation: ",o1.corr())
plt.scatter(o1["age"],o1["circumference"], alpha=0.5)
plt.show()