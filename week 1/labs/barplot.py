import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
########
m1=pd.read_csv("mtcars.csv")
cf = m1["cyl"].value_counts()
cd=cf.apply(pd.Series)
cd.columns=["freq"]
print(cd)
plt.bar(cd.index,cd["freq"])
plt.show()