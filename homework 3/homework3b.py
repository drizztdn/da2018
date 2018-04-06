# import the library
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.api as sma
from appJar import gui

pd.options.mode.chained_assignment = None

def train(trn):
    res = sm.glm(formula="Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5",data=trn,family=sma.families.Binomial()).fit()
    print(res.summary())
    return res

def predict(trn, smpl):
    ypnew = trn.predict(smpl)
    smpl['pred'] = ypnew
    smpl['pred_dir'] = ''
    smpl['pred_dir'][smpl['pred'] > .5] = "Down"
    smpl['pred_dir'][smpl['pred'] <= .5] = "Up"
    print(smpl)
    return smpl

df=pd.read_csv("smarket.csv")
rgs = train(df)

# create a GUI variable called app
app = gui()
app.addLabel("title", "Welcome to My Regression. I will predict the stock market")
app.setLabelBg("title", "red")
app.addLabelEntry("Lag1")
app.addLabelEntry("Lag2")
app.addLabelEntry("Lag3")
app.addLabelEntry("Lag4")
app.addLabelEntry("Lag5")
app.addLabel("Direction", "Prediction Soon...")

def press(button):
    if button == "Cancel":
        app.stop()
    else:
        smpl = pd.DataFrame(columns=['Lag1','Lag2','Lag3','Lag4','Lag5'])
        smpl.loc[0] = [float(app.getEntry("Lag1")),float(app.getEntry("Lag2")),float(app.getEntry("Lag3")),float(app.getEntry("Lag4")),float(app.getEntry("Lag5"))]
        print(smpl)
        smpl = predict(rgs,smpl)
        app.setLabel("Direction",smpl['pred_dir'][0])
app.addButtons(["Submit", "Cancel"], press)
# start the GUI
app.go()