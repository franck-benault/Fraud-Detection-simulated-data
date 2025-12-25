
import pickle
import pandas as pd


def getTarget():
    return 'TX_FRAUD'  

def getPredictors(dataFrame):
    target=getTarget()
    predictors = [col for col in dataFrame.columns ]
    predictors.remove(target)
    predictors.remove('TX_DATETIME')
    predictors.remove('CUSTOMER_ID')
    predictors.remove('TX_FRAUD_SCENARIO')
    return predictors



def getDataLearningAndValidation():
    filesLearning =['2018-04-01.pkl','2018-04-02.pkl','2018-04-03.pkl','2018-04-04.pkl','2018-04-05.pkl','2018-04-06.pkl','2018-04-07.pkl']
    filesValidation=['2018-04-08.pkl','2018-04-09.pkl','2018-04-10.pkl']
    folder='../data/raw/'
    list= []
    for file in filesLearning:
        with open(folder+file, 'rb') as file:
            dftemp = pd.read_pickle(file)
            list.append(dftemp)

    dfLearning = pd.concat(list, ignore_index=True)

    list= []
    for file in filesValidation:
        with open(folder+file, 'rb') as file:
            dftemp = pd.read_pickle(file)
            list.append(dftemp)

    dfValidation = pd.concat(list, ignore_index=True)
    return dfLearning,dfValidation


    