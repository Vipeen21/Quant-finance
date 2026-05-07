This "Hybrid" approach combines the **academic prestige** of Option 1 with the **practical accessibility** of Option 2. It’s designed to impress recruiters with your theoretical knowledge while showing you actually know how to build and backtest tools.

***

# Quantitative Finance & Stochastic Calculus 📈

This repository bridges the gap between high-level stochastic theory and practical algorithmic implementation. It features comprehensive notebooks and scripts covering everything from foundational Black-Scholes models to the complexities of the Heston stochastic volatility framework.

---

## 🚀 Key Features

### 1. Stochastic Volatility & The Heston Model
Going beyond constant volatility to model market dynamics more accurately. This implementation includes:
* **Calibration**: Fitting the model to market data.
* **Pricing**: Using the Heston SDEs:
  $$dS_t = \mu S_t dt + \sqrt{\nu_t} S_t dW_{1,t}$$
  $$d\nu_t = \kappa(\theta - \nu_t)dt + \sigma \sqrt{\nu_t} dW_{2,t}$$

### 2. Numerical Methods & Greeks
* **Finite Difference Methods**: Solving Black-Scholes PDEs for European and exotic options.
* **Implied Volatility Surface**: Generating 3D visualizations of volatility smiles and skews.
* **Itô’s Lemma**: Practical application of stochastic calculus for derivative pricing.

### 3. Backtesting & Asset Analysis
* **Strategy Execution**: Using `backtesting.py` to run quantitative strategies.
* **Risk Management**: Risk-neutral pricing and market analysis scripts.

---

## 📁 Repository Structure

| Folder/File | Description |
| :--- | :--- |
| `getting_started_tutorials` | Introductory notebooks for Itô Calculus and basic finance. |
| `Heston Pricing.ipynb` | Deep dive into Stochastic Volatility modeling. |
| `the_implied_volatility_surface.ipynb` | Visualizing market sentiment across strikes/expiries. |
| `algo trading with backtesting.py` | Implementation of automated trading logic. |
| `itos_lemma.ipynb` | The mathematical backbone of the entire repository. |

---

## 🛠️ Tech Stack & Setup

* **Core Logic:** Python 3.x
* **Analysis:** NumPy, SciPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Backtesting:** Backtesting.py

### Installation
```bash
git clone https://github.com/Vipeen21/Quant-finance.git
cd Quant-finance
pip install -r requirements.txt # Or install numpy, scipy, matplotlib, backtesting
```

---

## 🧪 Quick Start
To see the power of stochastic calculus in action, I recommend starting with:
1.  **`itos_lemma.ipynb`**: To understand the underlying math.
2.  **`Black-ScholesTrading.ipynb`**: To see the theoretical model applied to trade.
3.  **`the_implied_volatility_surface.ipynb`**: For high-end data visualization.

---
*Maintained by [Vipeen Kumar](https://github.com/Vipeen21)*

***
