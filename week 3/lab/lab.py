import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.tools.eval_measures as em

default = pd.read_csv("default.csv")
mtcars = pd.read_csv("mtcars.csv")
smarket = pd.read_csv("smarket.csv")
