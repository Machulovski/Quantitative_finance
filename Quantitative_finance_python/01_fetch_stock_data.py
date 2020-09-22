import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt

stocks = ['002594.SZ']

start_date = '01/01/2018'
end_date = '31/08/2020'

data = web.DataReader(stocks, data_source='yahoo', start=start_date, end=end_date)['Adj Close']

daily_returns = (data/data.shift(1))-1

daily_returns.hist(bins = 100)
plt.savefig(str(stocks[0])+'.jpg', bbox_inches='tight',dpi=150)
plt.show()

