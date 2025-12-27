import pandas as pd


def load_performance_result():
    usecols = ['Package','Name','Hyperparameters','F1']
    data = pd.read_csv("../data/results/performanceF1.csv", usecols=usecols)
    data=data.sort_values(["Package", "Name","Hyperparameters"])
    return data

def save_performance_result(timeResponse): 
    timeResponse.to_csv('../data/results/performanceF1.csv', index=False) 

def update_performance_result(package, name,hyperparameters, f1):
    f1=round(f1,4)


    data = load_performance_result()
    
    res=data[(data['Package']==package) 
        & (data['Name']==name)
        & (data['Hyperparameters']==hyperparameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        data.loc[index, 'F1']=f1
    else:
        data=pd.concat([pd.DataFrame([[package,name,hyperparameters,f1]], columns=data.columns), data], ignore_index=True)

    save_performance_result(data)


