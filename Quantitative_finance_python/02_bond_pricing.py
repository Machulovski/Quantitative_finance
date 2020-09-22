from math import exp

def zero_bond_price(par_value, market_rate, n):
    return par_value/(1+market_rate)**n

def bond_price(par_value, coupon_rate, market_rate, n):
    c = par_value*coupon_rate
    return c/market_rate*(1-(1/(1+market_rate)**n)) + par_value/(1+market_rate)**n

if __name__ == "__main__":
    
    par_value = 1000
    coupon_rate = 0.05
    n = 3
    market_rate = 0.04

    print("Price of the zero-coupon bond: $%0.2f" % zero_bond_price(par_value, market_rate,n))
    print("Price of the coupon bond: $%0.2f" % bond_price(par_value, coupon_rate,market_rate,n))
