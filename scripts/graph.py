import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 



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