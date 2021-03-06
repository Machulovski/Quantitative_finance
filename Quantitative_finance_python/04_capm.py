import pandas_datareader as pdr
from pandas_datareader import data, wb
from datetime import date
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

risk_free_rate = 0.05

def capm(start_date, end_date, ticker1, ticker2):
    # get stock data
    stock1 = pdr.get_data_yahoo(ticker1, start_date, end_date)
    stock2 = pdr.get_data_yahoo(ticker2, start_date, end_date)
    # get monthly returns
    return_stock1 = stock1.resample('M').last()
    return_stock2 = stock2.resample('M').last()
    # create data frame with adj close price
    data = pd.DataFrame({'s_adjclose':return_stock1['Adj Close'],'m_adjclose' : return_stock2['Adj Close']}, index=return_stock1.index)
    # calculate and add log returns
    data[['s_returns','m_returns']] = np.log(data[['s_adjclose','m_adjclose']]/data[['s_adjclose','m_adjclose']].shift(1))
    data = data.dropna()
    #covariance matrix
    covmat = np.cov(data['s_returns'],data['m_returns'])
    
    beta = covmat[0,1]/covmat[1,1]
    print("Beta from formula: ", beta)

    beta, alpha = np.polyfit(data["m_returns"],data["s_returns"],deg =1 )
    print("Beta from regression: ", beta)
    #plot
    fig, axis = plt.subplots(1, figsize=(20,10))
    axis.scatter(data["m_returns"], data["s_returns"],label="Data points")
    axis.plot(data["m_returns"],beta*data["m_returns"]+alpha, color="red", label="CAPM LINE")
    plt.title('Capital Asset Pricing Model, finding alphas and betas')
    plt.xlabel('Market return $R_m$', fontsize=18)
    plt.ylabel('Stock return $R_a$')
    plt.text(0.08,0.05,r'$R_a = \beta * R_m + \alpha$',fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.savefig(str(ticker1)+'_CAPM.jpg', bbox_inches='tight',dpi=150)
    plt.show()

    expected_return = risk_free_rate + beta*(data["m_returns"].mean()*12-risk_free_rate)
    print("Expected return: ", expected_return)

if __name__ == "__main__":
    capm('2019-01-01','2020-08-31','TSLA', '^IXIC')
