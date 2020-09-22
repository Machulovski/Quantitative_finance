import numpy as np
import pandas as pd 
from scipy.stats import norm
import pandas_datareader.data as web
import datetime

# VaR for tomorrow
def value_at_risk(position, c, mu, sigma):
    alpha = norm.ppf(1-c)
    var = position*(mu-sigma*alpha)

    return var

# VaR in n days time
def value_at_risk_long(S,c,mu,sigma,n):
    alpha = norm.ppf(1-c)
    var = S*(mu*n-sigma*alpha*np.sqrt(n))

    return var

if __name__ == "__main__":
    
    start_date = datetime.datetime(2018,1,1)
    end_date = datetime.datetime(2020,9,15)

    stock = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)
    stock['returns'] = stock['Adj Close'].pct_change()

    S = 1e6
    c = 0.99
    n = 5
    mu = np.mean(stock['returns'])
    sigma = np.std(stock['returns'])

    print("Value at risk is : $%0.2f" % value_at_risk(S,c,mu,sigma))
    print("Value at risk in 5 days is : $%0.2f" % value_at_risk_long(S,c,mu,sigma, n))