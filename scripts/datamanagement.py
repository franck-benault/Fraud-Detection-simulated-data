
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

def show_confusion_matrix(y_test,y_pred,title):
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
    sns.heatmap(cm, 
    xticklabels=['Not Fraud', 'Fraud'],
    yticklabels=['Not Fraud', 'Fraud'],
    annot=True,ax=ax1,
    fmt='d',
    linewidths=.2,linecolor="Darkblue", cmap="Blues")
    cm.style.format("{:20}")
    plt.title(title, fontsize=14)
    plt.show()


def plt_train_test(range, tabf1Train,trainLabel="f1 Train",tabf1Test=[], testLabel="f1 test"):
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot()

    ax1.set_ylabel(trainLabel)
    ax1.plot(range, tabf1Train, color = 'red', label = trainLabel)
    ax1.legend(loc = 'upper left')

    if(len(tabf1Test)==len(tabf1Train)):
        ax2 = ax1.twinx()
        ax2.set_ylabel(testLabel)
        ax2.plot(range, tabf1Test, color = 'blue', label = testLabel)
        ax2.legend(loc = 'upper right')

    fig.autofmt_xdate()
    plt.show()
    