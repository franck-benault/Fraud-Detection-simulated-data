

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
    