import numpy as np
import pandas as pd 
from scipy.stats import norm
import pandas_datareader.data as web
import datetime
import math

class ValueAtRiskMonteCarlo():

    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations
    
    def simulation(self):

        stock_data = np.zeros([self.iterations,1])
        rand = np.random.normal(0,1,[1,self.iterations])

        stock_price = self.S * np.exp(self.n*(self.mu-0.5*self.sigma**2) + self.sigma*np.sqrt(self.n)*rand)
        stock_price = np.sort(stock_price)

        percentile = np.percentile(stock_price,(1-self.c)*100)

        return self.S - percentile

if __name__ == "__main__":
    
    S = 1e6
    c = 0.99
    n = 1
    iterations = 1000000

    start_date = datetime.datetime(2019,1,1)
    end_date = datetime.datetime(2020,8,31)

    stock = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)

    stock['returns'] = stock['Adj Close'].pct_change()

    mu = np.mean(stock['returns'])
    sigma = np.std(stock['returns'])

    model = ValueAtRiskMonteCarlo(S,mu,sigma,c,n,iterations)

    print("Value at risk with MCS: $%0.2f" %model.simulation())