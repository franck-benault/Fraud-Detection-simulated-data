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


##-------------------------
def load_time_response_result():
    usecols = ['Package','Name','Hyperparameters','Learning time']
    timeResponse = pd.read_csv("../data/results/timeResponse.csv", usecols=usecols)
    timeResponse=timeResponse.sort_values(["Package", "Name","Hyperparameters"])
    return timeResponse

def save_time_response_result(timeResponse): 
    timeResponse.to_csv('../data/results/timeResponse.csv', index=False) 

def update_time_response_result(package, name,hyperparameters, learningTime):
    learningTime=int(learningTime)
    #learningTime=round(learningTime,1)

    timeResponsePandas = load_time_response_result()
    
    res=timeResponsePandas[(timeResponsePandas['Package']==package) 
        & (timeResponsePandas['Name']==name)
        & (timeResponsePandas['Hyperparameters']==hyperparameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        timeResponsePandas.loc[index, 'Learning time']=learningTime
    else:
        timeResponsePandas=pd.concat([pd.DataFrame([[package,name,hyperparameters,learningTime]], columns=timeResponsePandas.columns), timeResponsePandas], ignore_index=True)

    save_time_response_result(timeResponsePandas)

##-------------------
def load_learning_test_result():
    usecols = ['Package','Name','Hyperparameters','diff','F1Learning','F1Test']
    timeResponse = pd.read_csv("../data/results/learnngTest.csv", usecols=usecols)
    timeResponse=timeResponse.sort_values(["Package", "Name","Hyperparameters"])
    return timeResponse

def save_learning_test_result(timeResponse): 
    timeResponse.to_csv('../data/results/learnngTest.csv', index=False) 

def update_learning_test_result(package, name, hyperparameters, F1Learning, F1Test):
    #learningTime=int(learningTime)
    F1Learning=round(F1Learning,3)
    F1Test=round(F1Test,3)
    diff=round(F1Learning-F1Test,3)

    timeResponsePandas = load_learning_test_result()
    
    res=timeResponsePandas[(timeResponsePandas['Package']==package) 
        & (timeResponsePandas['Name']==name)
        & (timeResponsePandas['Hyperparameters']==hyperparameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        timeResponsePandas.loc[index, 'diff']=diff
        timeResponsePandas.loc[index, 'F1Learning']=F1Learning
        timeResponsePandas.loc[index, 'F1Test']=F1Test
    else:
        timeResponsePandas=pd.concat([pd.DataFrame([[package,name,hyperparameters,diff,F1Learning,F1Test]], columns=timeResponsePandas.columns), timeResponsePandas], ignore_index=True)

    save_learning_test_result(timeResponsePandas)

##-------------------

def load_hyperparameters_result():
    usecols = ['Package','Name','Hyperparameters','Scaler','values']
    timeResponse = pd.read_csv("../data/results/hyperparameters.csv", usecols=usecols)
    timeResponse=timeResponse.sort_values(["Package", "Name","Hyperparameters"])
    return timeResponse

def save_hyperparameters_result(timeResponse): 
    timeResponse.to_csv('../data/results/hyperparameters.csv', index=False) 

def update_hyperparameters_result(package, name, hyperparameters, values, scaler): 
    timeResponsePandas =  load_hyperparameters_result()
    
    res=timeResponsePandas[(timeResponsePandas['Package']==package) 
        & (timeResponsePandas['Name']==name)
        & (timeResponsePandas['Hyperparameters']==hyperparameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        print('trace')
        timeResponsePandas.loc[index, 'values']=str(values)
        timeResponsePandas.loc[index, 'Scaler']=str(scaler)
    else:
        timeResponsePandas=pd.concat([pd.DataFrame([[package,name,hyperparameters,values]], columns=timeResponsePandas.columns), timeResponsePandas], ignore_index=True)

    save_hyperparameters_result(timeResponsePandas)