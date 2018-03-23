import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
######
o = pd.read_csv("orange.csv")
o1 = o[["age","circumference"]]
print(o1)
plt.boxplot(o1["age"])
plt.show()