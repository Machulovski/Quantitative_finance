from math import exp

def discrete_future_value(x,r,n):
    return x*(1+r)**n

def discrete_present_value(x,r,n):
    return x*(1+r)**-n

def continuous_future_value(x,r,t):
    return x*exp(r*t)

def continuous_present_value(x,r,t):
    return x*exp(-r*t)

if __name__ == "__main__":
    
    x = 100
    r = 0.05
    n = 5

    print("Future value of $"+str(x)+" in "+str(n)+" year(s)(discrete model):", discrete_future_value(x,r,n))
    print("Future value of $"+str(x)+" in "+str(n)+" year(s)(continuous model):", continuous_future_value(x,r,n))
    print("Present value of $"+str(x)+" in "+str(n)+" year(s)(discrete model):", discrete_present_value(x,r,n))
    print("Future value of $"+str(x)+" in "+str(n)+" year(s)(continuous model):", continuous_present_value(x,r,n))