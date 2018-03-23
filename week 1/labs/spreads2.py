import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#######
q1=pd.read_csv("quakes.csv")
print(q1["mag"].describe())
print(q1.describe())