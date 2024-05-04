import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp


def optimize_rmin(R,C,R_min): 
    """
    minimize portfolio variance subject to R>=R_min.
    R: Return vector
    C: Covariance matrice
    R_min: constraint return
    """
    x = cp.Variable(R.shape[0],nonneg=True)
    #P = C
    objective = cp.Minimize(cp.quad_form(x, C))
    constraints = [x.T@R>=R_min,cp.sum(x)==1.0]
    prob = cp.Problem(objective,constraints)
    prob.solve()
    return x

def plot_efficient_frontier(R,C,N):
    """
    Plot efficient frontier.
    R: Return vector
    C: Covariance matrice
    N: number of portfolios
    """
    R_min_range = np.linspace(min(R),max(R),N)
    portfolio_std = []
    portfolio_R = []
    for r in R_min_range:
        try:
            arr = optimize_rmin(R,C,r).value
            if isinstance(arr,np.ndarray):
                portfolio_R.append(arr.T@R)
                portfolio_std.append(np.sqrt(arr.T@C@arr))
        except:
            pass
    plt.scatter(portfolio_std,portfolio_R,color="blue")
    plt.xlabel("std")
    plt.ylabel("Return")

def sharpe(x,rf,C,R):
    """
    minimize portfolio variance subject to R>=R_min.
    x: portfolio weights
    rf: risk-free rate
    C: Covariance matrice
    R: Return vector
    """
    return (x@R-rf)/np.sqrt(x.T@C@x)




def find_market_portfolio(R,C,rf,N_iter=10,epsilon=None):
    """
    find market risky portfolio (excluding rf) regarding a rf value.
    R: Return vector
    C: Covariance matrice
    rf: risk-free rate
    N_iter: number of iterations
    epsilon: small value for binary search adjustment
    """
    
    if not epsilon:
        epsilon=0.00000001+C.min()/100
    
    up,down=max(R),min(R)
    mid = (up+down)/2
    
    sharpe_up,sharpe_down,sharpe_mid = 0,0,0
    
    for _ in range(N_iter):
        try:
            sharpe_up = sharpe(optimize_rmin(R,C,up).value,rf,C,R)
            sharpe_down = sharpe(optimize_rmin(R,C,down).value,rf,C,R)
            sharpe_mid = sharpe(optimize_rmin(R,C,mid).value,rf,C,R)
            if sharpe_up>sharpe_down:
                down = (down+mid)/2 -epsilon
            else:
                up = (up+mid)/2 + epsilon
            mid = (up+down)/2
        except Exception as e:
            print("exception:",e)
    return optimize_rmin(R,C,mid).value