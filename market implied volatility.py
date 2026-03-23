import numpy as np
import qfin
from scipy.optimize import least_squares
from scipy.optimize._lsq.least_squares import call_minpack

S = 100
K = 100
r = 0.05
T = 1
sigma = 0.1 

euro_call = qfin.options.BlackScholesCall(S,sigma,K,r,T)
euro_put = qfin.options.BlackScholesPut(S,sigma,K,r,T)
print("euro call:", euro_call.price, "euro put:", euro_put.price)
print("euro call delta:", euro_call.delta, "euro put delta:", euro_put.delta)
print("euro call gamma:", euro_call.gamma, "euro put gamma:", euro_put.gamma)
print("euro call theta:", euro_call.theta, "euro put theta:", euro_put.theta)
print("euro call vega:", euro_call.vega, "euro put vega:", euro_put.vega)

#let option price be 10
call_option_price = 10

def diff(sigma):
    return np.abs(qfin.options.BlackScholesCall(S,sigma,K,r,T).price - call_option_price)
print("difference:", diff(.865555555))
result = least_squares(diff, sigma)
print("implied volatility:", np.round(result.x, 5 ))

C= qfin.options.BlackScholesCall(S,np.round(result.x, 5 ),K,r,T)
print("call price:", C.price)


