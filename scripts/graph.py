import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np



def show_importance(modelClf,predictors):
    if(hasattr(modelClf,"feature_importances_")):
        tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': modelClf.feature_importances_})
        tmp = tmp.sort_values(by='Feature importance',ascending=False)
        plt.figure(figsize = (7,4))
        plt.title('Features importance',fontsize=14)
        s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
        plt.show() 
    else:
        print("No feature importance")


def show_prediction_graph(modelClf, x_test,y_test):
    prediction=modelClf.predict_proba(x_test)[:,1]
    plt.figure(figsize=(10,5))
    list =np.array([])
    list0=prediction[y_test==0]
    list1=prediction[y_test==1]
    plt.hist(list0, bins=20, label='Negatives',alpha=0.5)
    for i in np.arange(0,len(list0)/len(list1)): 
        list= np.append(list,list1)

    plt.hist(list, bins=20, label='Positives', alpha=0.5, color='r')
    plt.xlabel('Probability of being Positive Class', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
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