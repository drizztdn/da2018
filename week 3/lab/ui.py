# import the library
from appJar import gui
# create a GUI variable called app
app = gui()
app.addLabel("title", "Welcome to My Regression. I will predict heart weight!")
app.setLabelBg("title", "red")
app.addLabelEntry("Type")
app.addLabelEntry("BodyWeight")
app.addLabel("Heart", "Heart weight. Soon...")

def press(button):
    if button == "Cancel":
        app.stop()
    else:
        type = app.getEntry("Type")
        weight = app.getEntry("BodyWeight")
        print("Type:", type, "Body Weight:", weight)
        app.setLabel("Heart",int(weight)*3)
app.addButtons(["Submit", "Cancel"], press)
# start the GUI
app.go()