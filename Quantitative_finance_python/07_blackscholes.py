import numpy as np
import pandas as pd 
from numpy import log, exp, sqrt
from scipy import stats

def blackscholes_call(S,E,T,rf,sigma):
    d1 = (log(S/E)+(rf + sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    return S*stats.norm.cdf(d1)-E*exp(-rf*T)*stats.norm.cdf(d2)

def blackscholes_put(S,E,T,rf,sigma):
    d1 = (log(S/E)+(rf + sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    return -S*stats.norm.cdf(-d1)+E*exp(-rf*T)*stats.norm.cdf(-d2)

if __name__ == "__main__":
    
    S0 = 100
    E = 100
    T = 1
    rf = 0.05
    sigma = 0.2

    print("Call option price according to BSM: ", blackscholes_call(S0, E, T, rf, sigma))
    print("Put option price according to BSM: ", blackscholes_put(S0, E, T, rf, sigma))