import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as sco

stocks = ['AAPL','TSLA','AMZN']
number_of_assets = len(stocks)
rf = 0.025

start_date = '01/01/2019'
end_date = '31/08/2020'

def download_data(stocks):
    data = web.DataReader(stocks, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
    data.columns = stocks
    return data

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

def calculate_returns(data):
    returns = np.log(data/data.shift(1))
    return returns

def plot_daily_returns(returns):
    returns.plot(figsize=(10,5))
    plt.show()

def show_statistics(returns):
    print(returns.mean()*252)
    print(returns.cov()*252)

def initialize_weights():
    weights = np.random.random(number_of_assets)
    weights /= np.sum(weights)
    return weights

def calculate_portfolio_return(returns,weights):
    portfolio_return = np.sum(returns.mean()*weights)*252
    print("Expected portfolio return: ", portfolio_return)

def calculate_portfolio_volatility(returns,weights):
    portfolio_volatility = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
    print('Expected portfolio volatility: ',portfolio_volatility)

def generate_portfolios(weights, returns):
    preturns = []
    pvolatility = []
    iterations = 10000

    for i in range(iterations):
        weights = np.random.random(number_of_assets)
        weights /= np.sum(weights)
        preturns.append(np.sum(returns.mean()*weights)*252)
        pvolatility.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252,weights))))
    
    preturns = np.array(preturns)
    pvolatility = np.array(pvolatility)

    return preturns, pvolatility

def plot_portfolios(preturns, pvolatility):
    plt.figure(figsize=(10,6))
    plt.scatter(pvolatility,preturns,c=(preturns-rf)/pvolatility,marker='o')
    plt.grid(True)
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()

def statistics(returns, weights):
    portfolio_return = np.sum(returns.mean()*weights)*252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252,weights)))

    return np.array([portfolio_return,portfolio_volatility,(portfolio_return-rf)/portfolio_volatility])

def min_func_sharpe(returns, weights):
    return -statistics(weights,returns)[2]

def optimize_portfolio(weights, returns):
    constraints = ({'type':'eq','fun':lambda x: np.sum(x)-1})
    bounds = tuple((0,1) for x in range(number_of_assets))
    optimum = sco.minimize(fun=min_func_sharpe,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraints)
    return optimum

def print_optimal_portfolio(optimum, returns):
    print("Optimal weights:", optimum['x'].round(3))
    print("Expected return, volatility and sharpe ratio:", statistics(returns,optimum['x'].round(3)))

def show_optimal_portfolio(optimum, returns, preturns, pvolatility):
    plt.figure(figsize=(10,6))
    plt.scatter(pvolatility,preturns,c=(preturns-rf)/pvolatility,marker='o')
    plt.grid(True)
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.plot(statistics(returns, optimum['x'])[1], statistics(returns, optimum['x'])[0],'g*', markersize=20.0)
    plt.show()

if __name__ == "__main__":
    data = download_data(stocks)
    show_data(data)
    returns = calculate_returns(data)
    plot_daily_returns(returns)
    show_statistics(returns)
    weights = initialize_weights()
    calculate_portfolio_return(returns, weights)
    calculate_portfolio_volatility(returns, weights)
    preturns, pvolatility = generate_portfolios(weights,returns)
    plot_portfolios(preturns,pvolatility)
    optimum = optimize_portfolio(weights,returns)
    print_optimal_portfolio(optimum,returns)
    show_optimal_portfolio(optimum,returns,preturns,pvolatility)

