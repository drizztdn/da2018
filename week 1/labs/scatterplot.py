import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

q1=pd.read_csv("quakes.csv")
print(q1)
plt.scatter(q1["lat"], q1["long"], alpha=0.5)
plt.show()