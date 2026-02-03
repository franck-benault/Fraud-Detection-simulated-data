from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler


def getScalers():
    scalers={"StandardScaler":StandardScaler(),
            "MinMaxScaler":MinMaxScaler(), 
            "RobustScaler":RobustScaler(),
            "MaxAbsScaler":MaxAbsScaler()}
    return scalers

def getScaler(key):
    return getScalers().get(key)