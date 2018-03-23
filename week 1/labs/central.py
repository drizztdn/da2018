import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##########
d = pd.Series([5,7,2,4,5,6,4,5,6,4,3,5,6,5,3])
print("Mean: ",d.mean())
print("Mode: ", d.mode())
print("Median: ", d.median())