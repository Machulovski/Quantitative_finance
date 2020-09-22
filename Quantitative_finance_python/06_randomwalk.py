import math
import numpy as np
import numpy.random as npr
import scipy
import matplotlib.pyplot as plt

def brownian_motion(dt=0.1,X0=0,N=1000):
    
    W = np.zeros(N+1)
    t = np.linspace(0,N,N+1)
    W[1:N+1] = np.cumsum(npr.normal(0,dt,N))

    return t, W

def plot_brownian_motion(t,W):
    plt.plot(t,W)
    plt.xlabel('Time(t)')
    plt.ylabel('Wiener-process W(t)')
    plt.title('Wiener-process')
    plt.show()

if __name__ == "__main__":
    t, W = brownian_motion()
    plot_brownian_motion(t,W)