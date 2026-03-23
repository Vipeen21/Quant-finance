import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import ljust
import scipy as sc
import pandas as pd
from scipy import stats
import QuantLib as ql

#initialise parameters
S0 = 100 # intial stock price
K = 150 #strike price
r = 0.05 #annual risk free rate
tau = 1 #time to maturity in years

 #heston dependent parameters
v0 = 0.20**2 #initial variance under risk neutral dynamice
theta = 0.20**2 #long term mean of variance under risk neutral dynamics
kappa = 3 #rate of mean reversion under risk neutral dynamice
sigma = 0.2 #volatility of volatility
rho = 0.98 #correlation between returns and variance under risk neutral dynamics
lambd = 0 #risk premium of variance

#check this condition to avoid negative volatility in heston model
if (2*kappa*theta > sigma**2):
    print("condition satisfied", 2*kappa*theta, ">", sigma**2 )  
else: 
    print("condition not satisfied", 2*kappa*theta, "<", sigma**2)

#heston characteristic equation
def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    #constants
    a = kappa*theta
    b= kappa*lambd

    #common terms with repect to phi
    rspi = rho*sigma*phi*1j

    #define b parameter given phi and b
    d = np.sqrt((rspi - b)**2 +(phi*1j + phi**2)*sigma**2)

    #define g parameter given phi, b and d
    g= (b-rspi+d)/(b-rspi-d)

    #calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*tau)
    term2 = S0**(phi*1j)*((1-g*np.exp(d*tau))/(a-g))**(-2*a/sigma**2)
    exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*((1-np.exp(d*tau))/(1-g*np.exp(d*tau))))

    return exp1*term2*exp2

#rectangular integration
def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    P, umax, N = 0,100, 650
    dphi = umax/N #dphi is width
    for j in range(1,N):
        #rectangular integration
        phi = dphi * (2*j +1)/2 #midpoint to calculate height
        numerator = heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator
    return np.real((S0- K*np.exp(-r*tau))/2 + P/np.pi)

strikes = np.arange(60,180,1.0)
option_prices = heston_price_rec(S0, strikes, v0, kappa, theta, sigma, rho, lambd, tau, r) 

print(option_prices)

#second order finite difference approximation
prices = pd.DataFrame([strikes, option_prices]).transpose()
prices.columns = ['strike', 'price']
prices['curvature'] = (-2*prices['price'] + prices['price'].shift(1) + prices['price'].shift(-1))/1**2

#plotting
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylabel('call price ($)')
ax2 = ax.twinx()

ax.plot(strikes, option_prices, label='option prices')
ax2.plot(prices['strike'], prices['curvature'], label='$d^2C/dK^2 (\sim pdf)$', color= 'orange' )

ax.legend(loc='center right')
ax2.legend(loc= "upper right")
plt.xlabel('strikes (K)')
plt.ylabel('$f_\\tau(K)$')
plt.title("Risk-neutral PDF, $f_\mathbb{Q}(K, \\tau)$")
plt.show()

#comparing it with quantlib heston model

today = ql.Date(13, 9, 2025)
expiry_date = today + ql.Period(int(365*tau), ql.Days)

# Setting up discount curve
risk_free_curve = ql.FlatForward(today, r, ql.Actual365Fixed())
flat_curve = ql.FlatForward(today, 0.0, ql.Actual365Fixed())
riskfree_ts = ql.YieldTermStructureHandle(risk_free_curve)
dividend_ts = ql.YieldTermStructureHandle(flat_curve)

heston_process = ql.HestonProcess(riskfree_ts, dividend_ts, ql.QuoteHandle(ql.SimpleQuote(S0)), v0, kappa, theta, sigma, rho)
heston_model = ql.HestonModel(heston_process)
heston_handle = ql.HestonModelHandle(heston_model)
heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)

# Now doing some pricing and curvature calculations

vols = [heston_vol_surface.blackVol(tau, x) for x in strikes]

option_prices1 = []

for strike in strikes:
    option = ql.EuropeanOption( ql.PlainVanillaPayoff(ql.Option.Call, strike), ql.EuropeanExercise(expiry_date))

    heston_engine = ql.AnalyticHestonEngine(heston_model)
    option.setPricingEngine(heston_engine)

    option_prices1.append(option.NPV())

print(option_prices1)
prices = pd.DataFrame([strikes, option_prices, option_prices1]).transpose()
prices.columns = ['strike', 'Rectangular Int', 'QuantLib']
prices['curvature'] = (-2 * prices['QuantLib'] + prices['QuantLib'].shift(1) + prices['QuantLib'].shift(-1)) / 0.01**2


# And plotting...
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylabel('Call Price ($)')
ax2 = ax.twinx()

#ax.plot(strikes, vols, label='Black Vols')
ax.plot(strikes, option_prices1, label='Option Prices')
ax2.plot(prices['strike'], prices['curvature'], label='$d^2C/dK^2 (\sim ~pdf)$', color='orange')

ax.legend(loc="center right")
ax2.legend(loc="upper right")
plt.xlabel('Strikes (K)')
plt.ylabel('$f_\\tau(K)$')
plt.title('QuantLib: Risk-neutral PDF, $f_\mathbb{Q}(K, \\tau)$')
plt.show()

#differences between the two
mse = np.mean( (option_prices - option_prices1)**2)
print("QuantLib vs. Our Rect Int \n Mean Squared Error: ", mse)
prices.dropna()
prices.head(40)

inter = prices.dropna()
pdf = sc.interpolate.interp1d(inter.strike, np.exp(r*tau)*inter.curvature, kind = 'linear')
pdfc = sc.interpolate.interp1d(inter.strike, np.exp(r*tau)*inter.curvature, kind = 'cubic')

strikes= np.arange(61,179,1.0)

plt.plot(strikes,pdfc(strikes), '-+', label= 'cubic')
plt.plot(strikes, pdf(strikes), label='linear')
plt.xlabel('Strikes (K)')
plt.ylabel('$f_\\tau(K)$')
plt.title('Risk-neutral Pdf: $f_\mathbb{Q}(K,\\tau)$')
plt.legend()
plt.show()

#cumulative distribution function
cdf= sc.interpolate.interp1d(inter.strike, np.cumsum(pdf(strikes)), kind= 'linear')
plt.plot(strikes, cdf(strikes))

plt.xlabel('Strikes (K)')
plt.ylabel('$F_\\tau(K)$')
plt.title('Risk-neutral CDF: $F_\mathbb{Q}(K, \\tau)$')
plt.show()

#using risk neutral pdf to price complex derivatives
def integrand_call(x,K):
    return (x-K)*pdf(x)
def integrand_put(x,K):
    return (K-x)*pdf(x)

calls, puts = [], []
for K in strikes:
    # integral from K to infinity (looking at CDF, 179 Last defined value on range)
    call_int, err = sc.integrate.quad(integrand_call, K, 178, limit=1000, args=K)
   
    # integral from -infinity to K (Looking at CDF, 61 Lowest defined value on range)
    put_int, err = sc. integrate.quad(integrand_put, 61, K, limit=1000, args=K)
   
    call = np.exp(-r*tau) *call_int
    calls.append(call)

    # put-call parity
    put = put_int
    puts.append(put)

rnd_prices = pd.DataFrame([strikes, calls, puts]).transpose()
rnd_prices.columns = ['strike', 'Calls', 'Puts']
print(rnd_prices)
