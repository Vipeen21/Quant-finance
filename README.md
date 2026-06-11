# 📈 Quantitative Finance & Stochastic Calculus Engine

[![GitHub follow](https://img.shields.io/github/followers/Vipeen21?label=Follow%20%40Vipeen21&style=for-the-badge&color=orange)](https://github.com/Vipeen21)
[![GitHub stars](https://img.shields.io/github/stars/Vipeen21/Quant-finance?style=for-the-badge&color=yellow)](https://github.com/Vipeen21/Quant-finance/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Vipeen21/Quant-finance?style=for-the-badge&color=blue)](https://github.com/Vipeen21/Quant-finance/network/members)
[![GitHub license](https://img.shields.io/github/license/Vipeen21/Quant-finance?style=for-the-badge&color=green)](https://github.com/Vipeen21/Quant-finance/blob/main/LICENSE)


<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter Notebook">
</p>

This repository bridges the gap between high-level stochastic calculus theory and practical computational execution. It features interactive implementations covering everything from classical risk-neutral pricing frameworks to complex, non-constant volatility regimes and mathematical integration foundations.

---

## 🏗️ Engine Architecture & Mathematical Workflow

The core framework decouples foundational stochastic calculus elements from numerical layers and empirical calibration matrices to map complex asset profiles dynamically.

```mermaid
graph TB
    %% Class Definitions for Styling
    classDef titleStyle fill:none,stroke:none,font-size:28px,font-weight:bold,font-family:Arial;
    classDef sectionTitle fill:none,stroke:none,font-size:16px,font-weight:bold,font-family:Arial;
    classDef noisyInput fill:#FADBD8,stroke:#E74C3C,stroke-width:1.5px,rx:5px,ry:5px,font-family:Arial;
    classDef queryInput fill:#EBDEF0,stroke:#8E44AD,stroke-width:1.5px,rx:5px,ry:5px,font-family:Arial;
    classDef darkLayer fill:#5D6D7E,stroke:#34495E,stroke-width:1.5px,rx:4px,ry:4px,color:white,font-weight:bold,font-family:Arial;
    classDef whiteLayer fill:#FFFFFF,stroke:#34495E,stroke-width:1.5px,rx:4px,ry:4px,font-family:Arial;
    classDef intermediateText fill:#F2F4F4,stroke:#BDC3C7,stroke-width:1.5px,rx:4px,ry:4px,color:#7F8C8D,font-family:Arial;
    classDef activeText fill:#FFFFFF,stroke:#34495E,stroke-width:1.5px,rx:4px,ry:4px,color:black,font-weight:bold,font-family:Arial;
    classDef canvasOutput fill:#48C9B0,stroke:#16A085,stroke-width:1.5px,rx:4px,ry:4px,font-weight:bold,font-family:Arial;
    classDef canvasOutputHighlight fill:#F1948A,stroke:#C0392B,stroke-width:1.5px,rx:4px,ry:4px,font-weight:bold,font-family:Arial;

    %% Title
    TITLE[DiffusionGemma]:::titleStyle

    %% --- FIRST ROW: INPUT HEADERS ---
    subgraph Row1 [ ]
        direction LR
        NC_TITLE[Noisy Canvas]:::sectionTitle
        space1[ ]:::titleStyle
        IQ_TITLE[Input Query]:::sectionTitle
    end
    style Row1 fill:none,stroke:none

    %% --- SECOND ROW: INPUT TOKENS ---
    subgraph Row2 [ ]
        direction LR
        %% Noisy Canvas Inputs
        n1[pen]:::noisyInput
        n2[sea]:::noisyInput
        n3[test]:::noisyInput
        n4[hi]:::noisyInput
        
        %% Spacing
        space2[ ]:::titleStyle
        
        %% Input Query Inputs
        q1[The]:::queryInput
        q2[LLM]:::queryInput
        q3[picks]:::queryInput
    end
    style Row2 fill:none,stroke:none

    %% --- THIRD ROW: CORE ARCHITECTURE ---
    
    %% Left Top Block (First Denoising Stage)
    subgraph Block1 [ ]
        direction TB
        tel1[Token Embedding Layer]:::darkLayer
        den1["Denoiser<br>(decoder with<br>bidirectional attention)"]:::whiteLayer
        lm1[LM Head]:::darkLayer
        tel1 --> den1 --> lm1
    end
    style Block1 fill:#A9CCE3,stroke:#2980B9,stroke-width:1px,rx:8px,ry:8px

    %% Right Block (Encoder Stage)
    subgraph Block2 [ ]
        direction TB
        tel2[Token Embedding Layer]:::darkLayer
        enc["Encoder<br>(decoder with<br>causal attention)"]:::whiteLayer
        tel2 --> enc
    end
    style Block2 fill:#A9CCE3,stroke:#2980B9,stroke-width:1px,rx:8px,ry:8px

    %% --- FOURTH ROW: INTERMEDIATE TOKENS ---
    subgraph Row4 [ ]
        direction LR
        it1[all]:::intermediateText
        it2[words]:::activeText
        it3[dogs]:::intermediateText
        it4[where]:::intermediateText
    end
    style Row4 fill:none,stroke:none

    %% --- FIFTH ROW: SECOND DENOISING BLOCK ---
    subgraph Block3 [ ]
        direction TB
        tel3[Token Embedding Layer]:::darkLayer
        den2["Denoiser<br>(decoder with<br>bidirectional attention)"]:::whiteLayer
        lm2[LM Head]:::darkLayer
        tel3 --> den2 --> lm2
    end
    style Block3 fill:#A9CCE3,stroke:#2980B9,stroke-width:1px,rx:8px,ry:8px

    %% --- SIXTH ROW: OUTPUT CANVAS ---
    subgraph Row6 [ ]
        direction LR
        out1[all]:::canvasOutput
        out2[words]:::canvasOutput
        out3[gemma]:::canvasOutputHighlight
        out4[once]:::canvasOutput
    end
    style Row6 fill:none,stroke:none
    
    OC_TITLE[Updated canvas]:::sectionTitle

    %% --- CONNECTIONS & FLOW ---
    
    %% Inputs to Top Blocks
    n1 & n2 & n3 & n4 --> Block1
    q1 & q2 & q3 --> Block2

    %% Cross-block KV-cache & Conditioning
    Block2 -- KV-cache --> Block1
    Block2 -- KV-cache ----> Block3
    Block1 -- self-<br>conditioning ----> tel3

    %% Intermediate Connections
    Block1 --> it1 & it2 & it3 & it4
    it1 & it2 & it3 & it4 --> Block3
    
    %% Final Output Connections
    Block3 --> out1 & out2 & out3 & out4
    out1 & out2 & out3 & out4 --- OC_TITLE

    %% Structural Alignment Tweaks
    TITLE --- Row1
    Row1 --- Row2
```


```mermaid
graph TD
    A[Stochastic Foundations: Ito & SDEs] --> B[Pricing Engine Frameworks]
    B --> C[Analytical: Black-Scholes]
    B --> D[Stochastic Volatility: Heston Model]
    B --> E[Numerical: Finite Difference Mesh]
    D --> F[Calibration Layer: Market Implied Vol Surfaces]
    E --> F
    F --> G[Dynamic Risk Mitigation & Volatility Analysis]

```

---

## 🎯 Core Frameworks Breakdown

### 1. Stochastic Volatility & The Heston Model

Real-world asset returns exhibit volatility clustering and leverage effects that classical constant-volatility frameworks fail to capture. This engine models asset dynamics via two coupled Stochastic Differential Equations (SDEs):

$$dS_t = \mu S_t dt + \sqrt{\nu_t} S_t dW_{1,t}$$

$$d\nu_t = \kappa(\theta - \nu_t) dt + \sigma \sqrt{\nu_t} dW_{2,t}$$

* **Calibration Layer:** Mathematically fits structural parameters ($\kappa, \theta, \sigma, \rho$) to empirical market option chains.
* **Pricing Engine:** Evaluates the characteristic function via Fourier inversions to value European derivatives under non-constant volatility.

### 2. Implied Volatility Surfaces & Numerical Discretization

* **Volatility Surfaces:** Generates dynamic, multi-dimensional structures mapping the continuous implied volatility smile and skew across varying strikes and maturities.
* **Finite Difference Methods:** Complements analytical equations by numerically solving pricing Partial Differential Equations (PDEs) under discrete boundary conditions.

---

## 📊 Comparative Framework Analysis

| Model Framework | Volatility Assumption | Solution Method | Key Strength | Ideal Use-Case |
| --- | --- | --- | --- | --- |
| **Black-Scholes** | Constant $\sigma$ | Analytical (Closed-form) | Speed & benchmark stability | Plain-vanilla liquid options |
| **Heston Model** | Stochastic $\nu_t$ (CIR Process) | Semi-Analytical (Fourier) | Captures smile, skew & leverage | Long-dated options, exotic profiles |
| **Finite Differences** | Flexible / Arbitrary | Numerical (Grid/Mesh) | Handles path-dependence & barriers | American options, custom boundaries |

---

## 📂 Repository Blueprint

```bash
├── getting_started_tutorials/     # Foundational concepts & entry points
├── Black-ScholesTrading.ipynb     # Closed-form pricing & basic Greeks infrastructure
├── Heston Pricing 1.ipynb         # SDE setup and characteristic function solving
├── Heston Pricing 2.ipynb         # Fourier inversions and parameters calibration
├── finite_differences_option_pricing.ipynb # PDE numerical mesh methods
├── the_implied_volatility_surface.ipynb    # 3D mapping of skew and smile curves
├── ito_integration.ipynb          # Computational notebooks on Ito's Integral
├── itos_lemma.ipynb               # Structural expansions of stochastic variables
├── market implied volatility.py   # Real-time asset implied volatility extraction
├── risk free option trading.py    # Arbitrage boundary verification scripts
└── LICENSE                        # Open-source distribution permissions

```

---

## 🗺️ Future Roadmap & Expansion Work

* [x] Implement core Ito Calculus & SDE simulation environments.
* [x] Build semi-analytical closed-form frameworks (Black-Scholes, Heston).
* [x] Launch 3D Implied Volatility Surface mapping toolsets.
* [ ] **Phase 4: Neural Volatility Operators:** Integrate Physics-Informed Neural Networks (PINNs) to accelerate option PDE grid solving under tight latency limits.
* [ ] **Phase 5: Rough Volatility Frameworks:** Implement fractional Brownian motion (e.g., Rough Heston, rBergomi) to address structural micro-market irregularities.
* [ ] **Phase 6: Advanced Calibration Optimization:** Deploy hybrid genetic-gradient algorithms to resolve non-convex objective spaces during Heston parameter fitting.

---

## 🤝 Community & Contribution

Whether you are looking to fix a mathematical edge-case, optimize a matrix calculation in NumPy, or add a new stochastic asset path generator, contributions are highly welcome!

1. **Fork** the project repository.
2. **Create** your feature branch (`git checkout -b feature/StochasticUpgrade`).
3. **Commit** your changes (`git commit -m 'Add neural network calibration layer'`).
4. **Push** to the branch (`git push origin feature/StochasticUpgrade`).
5. Open a **Pull Request**.

## 🙏 Connect & Collaborate

If this repository assists your quantitative research, trading strategy formulation, or stochastic calculus foundations, **consider giving it a star!** ⭐
[⭐ Star This Repo](https://github.com/Vipeen21/Quant-finance/stargazers) | [🍴 Fork This Repo](https://github.com/Vipeen21/Quant-finance/network/members)

**Vipeen Kumar** *Quantitative Researcher & Data Scientist*

Let's collaborate on quantitative finance, stochastic systems, and financial AI architecture.

* **Author:** Vipeen Kumar 
* **LinkedIn:** [Profile Link](https://www.google.com/search?q=https://linkedin.com/in/vipeen-kumar-908212b8)
* **Portfolio Website:** [vipeen21.github.io](https://www.google.com/search?q=https://vipeen21.github.io)

`#QuantitativeFinance` `#StochasticCalculus` `#HestonModel` `#AlgorithmicTrading` `#VolatilitySurface` `#OptionsPricing`
