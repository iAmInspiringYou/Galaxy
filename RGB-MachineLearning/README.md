<b> What is this for? </b>

A logistic regression classifier to classify between a dark color, and a light color. Can be used in a application where the text color has to change to a lighter color for a darker background, and vice-versa.

<b> Project contents </b>

RGBClassifier.py - To produce a neural network model for logistic regression to classify between the light/dark colors. 

Main.py - To use the learned model, and predict for new colors. Change the input on line number 11 - the numbers should be float, and between 0 and 1.
The color 255,255,255 is represented as 1.0,1.0,1.0 here, because every number is divided by 255.

model.ckpt - The learned model generated from RGBClassifier.py
