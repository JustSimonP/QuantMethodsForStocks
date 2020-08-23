import numpy as numpy
import os
import pandas as pandas
import random
import time# -*- coding: utf-8 -*-
import matplotlib.pyplot as plot
x = numpy.arange(1, 301)
y = numpy.random.normal(x+2, 50)

#plot.scatter(x,y)
#plot.show()

numerator = numpy.sum((x- numpy.mean(x)) * (y -numpy.mean(y)))
denominator = numpy.sum((x-numpy.mean(x)) **2)
b1 = numerator/denominator
b0 = numpy.mean(y) - b1 * numpy.mean(x)

def calc_predictions(x):
    return b0 + b1*x

y_preds = calc_predictions(x)
y_preds[:10]

def get_rmse(y_true, y_predicted):
    return numpy.sqrt(numpy.sum((y_predicted - y_true)**2)/len(y_true))

result = get_rmse(y, y_preds)
print(result)



plot.figure(figsize=(16,9))
plot.title('X vs Y Regression Line', fontsize=24, fontstyle='italic')
plot.scatter(x, y, s=256, alpha=0.5, label='Y')
plot.plot(numpy.arange(1,len(x)+1), sorted(y_preds),color='#e74c3c' ,linewidth=4,label='Linear: Predicted Y')
plot.show()


#OTHER POSSIBLE WAY
b1_short = numpy.corrcoef(x,y)[0][1] * (numpy.std(y)/numpy.std(x))
b0_short = numpy.mean(y) - b1_short * numpy.mean(x)




