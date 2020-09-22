import numpy as np
import pandas as pd 
from datetime import datetime
import sklearn
import pandas_datareader.data as web
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import confusion_matrix

def create_dataset(stock_symbol, start_date, end_date, lags = 5):
    df = web.DataReader(stock_symbol,"yahoo",start_date,end_date)
    tslag = pd.DataFrame(index=df.index)
    tslag['Today']=df['Adj Close']
    tslag['Volume']= df['Volume']

    for i in range(0, lags):
        tslag["lag%s" %str(i+1)] = df["Adj Close"].shift(i+1)
    
    dfret = pd.DataFrame(index=tslag.index)
    dfret['Volume'] = tslag['Volume']
    dfret['Today'] = tslag["Today"].pct_change()*100.0

    for i in range(0,lags):
        dfret["lag%s" %str(i+1)] = tslag["lag%s" %str(i+1)].pct_change()*100.0

    dfret['Direction'] = np.sign(dfret["Today"])

    dfret.drop(dfret.index[:6], inplace=True)

    return dfret

if __name__ == "__main__":
    data = create_dataset("TSLA",datetime(2013,1,1),datetime(2020,9,15),lags=5)

    X = data[["lag1","lag2",'lag3','lag4']]
    Y = data['Direction']
    #set test data
    start_test = datetime(2019,1,1)

    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    Y_train = Y[Y.index < start_test]
    Y_test = Y[Y.index >= start_test]

    model = SVC(C=1000000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0001, kernel='rbf', max_iter=-1, probability=False)
    # model.LinearSVC()

    model.fit(X_train,Y_train)

    pred = model.predict(X_test)

    print("Accuracy of SVM model: %0.3f" % model.score(X_test, Y_test))
    print("Confusion matrix: \n%s" % confusion_matrix(pred, Y_test))