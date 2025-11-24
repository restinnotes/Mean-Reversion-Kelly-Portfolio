# Single-Asset LEAPS Kelly Position Management Tool User Guide

> **Related Documentation**: For the mathematical principles and derivation process of this tool, please refer to the "Theoretical White Paper (PDF)".

This guide aims to instruct users on how to use `single_asset_kelly.py` (single-asset strategy script) to calculate optimal LEAPS positions based on mean reversion and the Kelly Criterion.

---

## 1. Core Philosophy

We don't predict whether stock prices will rise or fall tomorrow; instead, we calculate the current odds. This tool answers three core questions:

- **Drift**: According to the mean reversion model (OU Process), how strong is the stock's current rebound momentum?
- **Cost**: How much "rent" (Theta + cost of capital) do we need to pay to hold LEAPS leverage?
- **Size**: After deducting costs, if there's still positive edge (ERP), how much should we bet according to the Kelly Criterion?

---

## 2. Quick Start

### Environment Setup

Ensure your Python environment has the following libraries installed:

```bash
pip install numpy pandas yfinance
```

### Directory Structure

Ensure the script can access tools in the `code/utils/` folder:

```text
project_root/
├── code/
│   ├── strategies/
│   │   └── single_asset_kelly.py  # Main program
│   └── utils/
│       ├── lambda_tools.py        # Calculate reversion momentum
│       └── sigma_tools.py         # Calculate realized volatility
└── pe_csv/
    └── NVDA_pe.csv                # Historical PE data (for Lambda calculation)
```

### Run Script

```bash
python code/strategies/single_asset_kelly.py
```

---

## 3. Parameter Explanation: Objective Statistics vs. Subjective Judgment

This is the most critical part of the strategy. Model results depend on inputs, which are divided into two categories: **automatically obtained objective data** and **subjective parameters requiring your decision**.

### A. Automatic Acquisition (Objective Data) —— 📍 No Action Required

The program automatically calculates these statistical features from Yahoo Finance and local CSV files, representing the objective market state.

| Parameter | Code Variable | Meaning | Source |
|:---|:---|:---|:---|
| Reversion Speed | `lambda_annual` | Speed at which stock price reverts to mean after deviation. Higher values mean faster recovery and higher expected returns. | OU process fitting based on pe_csv data |
| Realized Volatility | `sigma_iv` | Annualized volatility of the underlying stock (e.g., 45%). This is the denominator (risk) in the Kelly formula. | yfinance historical data from past 3 years |
| Risk-free Rate | `r_f` | Opportunity cost of capital (e.g., 4.1%). | Hard-coded or API retrieval |

### B. Market Snapshot —— 📍 Manual Input Required

> ⚠️ **Critical Selection Rules**
>
> 1. Before entering the following data, mentally establish your **Hard Floor ($V_{hard}$)**.
> 2. The **Strike Price** of your chosen LEAPS contract should **equal or approximate** your hard floor.
>
> **Rationale**: Even if the stock price halves and falls below the hard floor, your maximum loss is capped by the strike price, physically truncating tail risk.

You need to open your trading software (e.g., IBKR, ThinkOrSwim), find the contract where Strike ≈ $V_{hard}$, and enter the data:

| Parameter | Code Variable | Description |
|:---|:---|:---|
| Stock Price | `P` | Current price of the underlying stock. |
| Option Price | `option_price` | Latest price (Ask Price) of your selected contract (Strike ≈ Hard Floor). |
| Delta | `delta` | Delta value of the contract (typically between 0.8 ~ 0.9). |
| Theta | `theta_daily_abs` | Daily Theta decay of the contract (absolute value). |

### C. Subjective Decisions (Subjective Inputs) —— 🎨 Your Art

This is where you exercise "investor insight". These parameters directly determine position aggressiveness.

#### 1. Target Price (V) and Hard Floor (V_hard)

```python
V = 225.00      # Target Price (Fair Value)
V_hard = 130.00 # Hard Floor --> Corresponds to LEAPS Strike Price
```

- **V (Target Price)**: What do you think the asset is worth? Could be previous high, DCF valuation, or average PE valuation.
  - **Impact**: Higher $V$ means higher expected return (Drift) and larger position.
- **V_hard (Hard Floor)**: In an extremely pessimistic scenario, the price you believe it absolutely won't fall below (e.g., historical minimum valuation support).
  - **Impact**: This determines your "confidence level". The closer the stock price is to $V_{hard}$, the higher the Alpha confidence coefficient the program gives, resulting in a heavier position.
  - **Action**: Be sure to purchase options with a strike price near this level.

#### 2. Valuation Discount Factor (beta)

```python
beta = 0.2  # Recommended range: 0.2 ~ 0.5
```

- **Meaning**: To what extent do you want to reduce position as stock price rises toward target price $V$?
- **Logic**:
  - **0.0**: Permabull mode. Even if stock reaches target price, Alpha remains 1.0, no position reduction.
  - **1.0**: Extremely conservative. Once stock reaches target, Alpha becomes 0, complete liquidation.
  - **0.2**: Recommended value. Maintains most of base position to guard against missing right-side momentum.

#### 3. Kelly Coefficient (k)

```python
k = 1.0  # Recommended range: 0.3 ~ 1.0
```

- **Meaning**: Safety margin multiplier.
- **Usage**: If calculation shows 120% position and you find it too aggressive, set $k$ to 0.5 (half-Kelly), result becomes 60%.
- **Practical Advice**: For single assets, recommend $k$ not exceeding 0.7; if highly confident with sufficient capital, can use 1.0.

---

## 4. Result Interpretation: How to Read the Report?

After running the script, you'll receive a bilingual report. Focus on these core metrics:

### Step 1: Check ERP (Net Edge)

> LEAPS Net Edge (ERP): 203.74% (after Rf & Theta)

This is the pure excess return after deducting interest and time decay.

- **If negative**: 🛑 Absolutely do not open position. Expected rebound momentum isn't enough to even cover the option's time rent.
- **If positive**: ✅ This is a mathematically positive expected value bet.

### Step 2: Check Alpha (Confidence Level)

> Current Level Risk: 54.9% (Alpha discount factor = 0.890)

This tells you the current stock price position within the [Hard Floor, Target] range.

- **0.890** means the program suggests deploying only 89% of theoretical position, leaving 11% margin for defense.

### Step 3: Check Suggested Position (Kelly Suggestion)

> Suggested Cash Investment: $122,506
> Suggested Contracts to Purchase: 18.95 contracts

This is the final conclusion.

- **Note**: If calculated ratio > 100% (as in this example), it represents excellent opportunity. But in practice, never use margin to buy options. Cap position at 100% or your account's preset single-asset limit (e.g., 20%).

---

## 5. Risk Warnings

- **Garbage In, Garbage Out (GIGO)**: If your $V$ (target price) is set too high, or $V_{hard}$ (hard floor) is set too low, calculation results will mislead you into heavy positions. Valuation is subjective; be responsible for yourself.

- **Black Swan Risk**: Kelly formula assumes historical volatility (Sigma) represents the future. If sudden collapse occurs, actual volatility may far exceed calculated values.

- **Leverage Double-Edged Sword**: Although LEAPS have low long-term decay, short-term volatility is enormous. Ensure your psychological tolerance can handle 50%+ drawdowns.

---

## 6. Real-World Case Analysis: Why Sharpe Ratio Isn't the Only Standard? (The Margin Trap)

When using `code/strategies/comparison_LEAPS_stock.py` for calculations, we may encounter a seemingly contradictory phenomenon: **the stock has a higher Sharpe ratio, but the program's suggested position is physically impossible to execute.**

Here's a set of real calculation data ($k=1.0$):

### Experimental Data Comparison

**Scenario A: Considering Real Theta Decay (Real World)**
| Metric | Stock | LEAPS | Winner |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | **1.7664** | 1.6746 | 🏆 Stock |
| **Suggested Position ($f$)** | **308.2%** | 132.7% | |
| **Verdict** | LEAPS' Theta cost drags down efficiency, stock wins. |

**Scenario B: Assuming Theta = 0 (Ideal World)**
| Metric | Stock | LEAPS | Winner |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | 1.7664 | **1.8131** | 🏆 LEAPS |
| **Suggested Position ($f$)** | 308.2% | 132.7% | |
| **Verdict** | After removing time decay, LEAPS' leverage advantage makes it 2.6% more efficient than stock. |

### Key Finding: The "Hidden Borrowing Cost" Trap of Stocks

Although in Scenario A, stocks seemingly won (Sharpe ratio 1.77 vs 1.67), this is not just a mathematical conclusion, but an **execution trap**.

1.  **Unexecutable Position**:
    The program suggests buying **308.2%** of stock. This means if you have $100K capital, you need to **borrow $200K margin** to buy stock. Even with half-Kelly ($k=0.5$), suggested position still reaches **154%**, still requiring margin.

2.  **Missing Borrowing Cost**:
    Current Sharpe ratio calculation **does not deduct margin interest for the stock portion**.
    * **Reality**: Broker margin rates typically reach 6%~8%. Adding this substantial cost would significantly reduce the stock's true ERP, causing Sharpe ratio to collapse instantly.
    * **LEAPS Advantage**: LEAPS come with built-in ~2.4x leverage, with leverage cost mainly reflected as Theta (already calculated) and embedded time value, **requiring no additional margin interest payment**.

### Final Conclusion: Why Only LEAPS?

In this case, although LEAPS' nominal Sharpe ratio is slightly lower (dragged by Theta), it has **irreplaceable practical advantages**:

* **Capital Efficiency**: LEAPS' suggested **132%** position ($k=1$) or **66%** position ($k=0.5$) is controllable in capital management. Especially at $k=0.5$, you only need to deploy 66% of capital to achieve approximately 1.6x ($66\% \times 2.4$) market exposure, and **completely margin-free**.
* **Avoiding Liquidation Risk**: Holding 300% stock position encountering 30% drawdown leads to account wipeout (liquidation); while holding LEAPS even if they zero out, losses are limited to premium paid (though painful, no debt liability).

**Therefore, when high-confidence opportunities arise, to obtain excess market exposure (Exposure > 100%) while avoiding margin interest, LEAPS is the only rational tool.**

---

## 7. Single-Asset Risk & Drawdown Analysis

After determining the Kelly-suggested position, risk stress testing is mandatory. We provide a dedicated script to answer one core question: **"If I buy at this position size and the market crashes tomorrow, how much will my account lose?"**

### 7.1 Running the Risk Analysis Script

Run the risk analysis script `single_asset_risk.py` located in the `code/strategies/` directory:

```bash
python code/strategies/single_asset_risk.py
```

### 7.2 Core Metrics Interpretation

The script output is divided into three key sections. It's crucial to distinguish between **instrument risk** and **account risk**:

#### A. Asset Characteristics (The Instrument)

* **Effective Leverage**:
  * Displayed as `L` times. For example, 2.4x means if the stock drops 1%, the option theoretically drops 2.4%.
  * *Note: This is dynamic; as stock price falls, leverage typically decreases, but Gamma risk increases.*
* **Kelly Suggested Position (Kelly Allocation)**:
  * Suggested cash allocation percentage based on `k` value.
  * If displays > 100%, the model is extremely bullish on this opportunity, suggesting full allocation (but in live trading, cap at 100% or manually reduce k value).

#### B. Daily Volatility

This is the key metric for assessing "psychological tolerance":

1. **LEAPS Instrument Daily Volatility (Instrument Risk)**
   * **Meaning**: Regardless of how much you buy, this is the average daily fluctuation amplitude of the option contract itself.
   * **Reference Value**: Tech stock LEAPS typically range **6% - 10%**. If it drops 7% one day, this is "normal volatility," no need to panic.

2. **Your Account Daily Risk (Account Risk)**
   * **Meaning**: After buying, how much will **your total account** (e.g., $100k) fluctuate daily.
   * **Formula**: `Account Volatility = Position Percentage × LEAPS Volatility`
   * **Practical Risk Thresholds**:
     * **< 2%**: Low risk (similar to bond funds/balanced portfolios).
     * **2% - 4%**: Aggressive growth (similar to holding NVDA stock at full allocation).
     * **> 5%**: **Extremely high risk** (similar to crypto futures). If you see this value, **you must reduce k value**.

### 7.3 Extreme Scenario Analysis

The script simulates three probability events based on normal distribution:

| Scenario | Probability | Meaning | Mental Response |
|:---|:---|:---|:---|
| **Normal Fluctuation (1σ)** | ~68% | Occurs once every 3 days | "Just normal market noise, close the app and sleep." |
| **Monthly Major Drop (2σ)** | ~95% | Occurs once a month on average | "Hurts a bit, but hasn't broken the uptrend, stay attentive." |
| **Extreme Crash (3σ)** | ~99% | Occurs once a year on average | "Black swan event. If loss exceeds psychological threshold, position is too heavy." |

### 7.4 How to Adjust Risk Control Parameters (k Value)

If you see a warning after running the script:

> `⚠️ High Risk Alert: Your account daily volatility (9.39%) is extremely high.`

Open the code file and find the following parameter to modify:

```python
# [Strategy Parameters]
k = 1.0   # Default aggressive mode (Full Kelly)
```

**Suggested adjustments:**

* **k = 0.5 (Half Kelly)**: Industry standard long-term survival configuration. Returns are 75% of full position, but volatility is halved. Suitable for most aggressive investors.
* **k = 0.3**: Conservative configuration. Suitable for investors who cannot psychologically tolerate account daily drawdown >3%.

---

**Comparison Example:**

* **k=1.0**: Position 120%, account daily volatility 9.4%. (Daily loss $9,400) -> **Unacceptable**
* **k=0.5**: Position 60%, account daily volatility 4.7%. (Daily loss $4,700) -> **Aggressive but controllable**

---

### 7.5 Practical Recommendations

For LEAPS strategies, **k=0.5 (Half Kelly)** is typically the optimal balance point for maximizing long-term geometric growth rate while avoiding ruin. Due to subjective factors (such as whether the target price is reasonable?) and the existence of black swans, this configuration can:

* Retain approximately 75% of theoretical returns
* Reduce volatility to 50% of the original
* Significantly improve psychological tolerance
* Reduce permanent capital loss risk from consecutive losses

**Remember**: The Kelly Criterion pursues long-term compound growth, not maximizing a single bet. Survival is the first priority.