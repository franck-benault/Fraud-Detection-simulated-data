import numpy as np
import pandas as pd

def generate(value,scale):
   return np.random.normal(loc=value, scale=scale, size=None)


def generateData(points, imbalanced,nbPoints=100,imbalanceRate=100):
    myList = []
    for i in range(0, nbPoints, 1):
        for p in points:
            if(p['fraudulent'] and imbalanced):
                if(i%imbalanceRate==0):
                    data1={"x":generate(p['x'],p['scalex']), "y":generate(p['y'],p['scaley']),"Fraudulent":p['fraudulent']}
                    myList.append(data1)
            else:
                data1={"x":generate(p['x'],p['scalex']), "y":generate(p['y'],p['scaley']),"Fraudulent":p['fraudulent']}
                myList.append(data1)
    return myList