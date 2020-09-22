import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.metrics import confusion_matrix

def create_dataset(stock_symbol, start_date, end_date, lags = 5):
    df = web.DataReader(stock_symbol,"yahoo",start_date,end_date)
    dflag = pd.DataFrame(index=df.index)
    dflag['Today']=df['Adj Close']
    dflag['Volume']= df['Volume']

    for i in range(0, lags):
        dflag["lag%s" %str(i+1)] = df["Adj Close"].shift(i+1)
    
    dfret = pd.DataFrame(index=dflag.index)
    dfret['Volume'] = dflag['Volume']
    dfret['Today'] = dflag["Today"].pct_change()*100

    for i in range(0,lags):
        dfret["lag%s" %str(i+1)] = dflag["lag%s" %str(i+1)].pct_change()*100

    dfret['Direction'] = np.sign(dfret["Today"])

    dfret.drop(dfret.index[:6], inplace=True)

    return dfret

if __name__ == "__main__":
    data = create_dataset("TSLA",datetime(2019,1,1),datetime(2020,9,15),lags=5)

    X = data[["lag1","lag2",'lag3','lag4']]
    Y = data['Direction']
    #set test data
    start_test = datetime(2020,1,1)

    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    Y_train = Y[Y.index < start_test]
    Y_test = Y[Y.index >= start_test]

    model = LogisticRegression()

    model.fit(X_train,Y_train)

    pred = model.predict(X_test)

    print("Accuracy of logstic regression model: %0.3f" % model.score(X_test, Y_test))
    print("Confusion matrix: \n%s" % confusion_matrix(pred, Y_test))