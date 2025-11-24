# 🌌 Merton-Kelly LEAPS Optimizer
### Quantitative Mean-Reversion · Dynamic LEAPS Portfolio Engine
### 量化均值回归 · 动态期权组合管理系统

---

[English](#english-version) / [中文](#中文版本)

---

## English Version <a name="english-version"></a>

### 📚 Documentation & Research

If you want to **set up, configure, and run** this quantitative framework, please consult the detailed **operational steps** in the:
* [**User Guide**](docs/md/UserGuide.md)

If you are interested in the **core mathematical framework, mean-reversion logic, Kelly formula** and **valuation adjustment** derivations, please read the:
* [**Whitepaper (PDF)**](docs/pdf/quantitative_cash_allocation_en.pdf)

---

### 📖 Why This Project Exists? (The Purpose)

We built this because **trading LEAPS by "gut feeling" is a nightmare.**
You either buy too much at the top and panic-sell, or buy too little at the bottom and miss the rally.

This project is a **mathematical anchor**. It answers one simple question:
> *"Given my conviction in this stock's value, exactly how much money should I risk right now to maximize growth without blowing up?"*

It turns vague "buy the dip" advice into a precise **Position Sizing Number**, powered by the Merton-Kelly formula.

---

### ⚙️ How It Works (The Logic)

1.  **Stop Guessing the Bottom:** We use **Mean Reversion**. The further the price falls below fair value, the harder the math pushes you to buy.
2.  **Respect the Volatility:** High-volatility stocks (like NVDA) are penalized. The model forces you to bet smaller on wild horses, keeping your sanity intact.
3.  **Leverage Efficiency:** We use **Deep ITM LEAPS** to control stock exposure with only ~40% capital, leaving 60% in cash (SGOV) as a safety net.

---

### 🚀 Core Features

* **Data-Driven:** Calculates $\lambda$ (Reversion Speed) & $\sigma$ (Volatility) from 3-year historical data.
* **Valuation-Adjusted:** Introduces a unique scalar ($C_{vol}$) that mathematically "discounts" risk when buying deep value.
* **Safety First:** Built-in **Half-Kelly** constraints and **Hard Cash Caps** to prevent over-leverage.

---

### ⚠️ Disclaimer

This is a quantitative research framework. Use responsibly.

---

## 中文版本 <a name="中文版本"></a>

### 📚 文档与原理参考

如果您想**配置、运行和使用**本量化框架，请查阅详细的**操作步骤**：
* [**使用指南**](docs/md/使用指南.md)

如果您对本项目的**数学原理、均值回归逻辑、凯利公式**及**估值修正**等底层细节感兴趣，请阅读：
* [**量化白皮书 (PDF)**](docs/pdf/quantitative_cash_allocation_zh.pdf)

---

### 📖 为什么要做这个项目？（初心）

我们做这个系统，是因为**凭感觉买 LEAPS 是一场噩梦。**
要么在山顶重仓然后心态崩盘，要么在谷底不敢买而踏空暴涨。

这个项目是一个**“数学锚点”**。它只为了解决一个终极问题：
> **“既然我看好这只股票的价值，我现在到底该买多少钱，才能既赚得快，又绝对不会爆仓？”**

它把模糊的“抄底”建议，转化为了一个精确的**仓位数字**。

---

### ⚙️ 它怎么帮你赚钱？（核心逻辑）

1.  **不再瞎猜底：** 利用**均值回归**原理。股价跌得越深，离估值越远，模型计算出的“赚钱动力”越大，给你的仓位建议就越重。
2.  **专治手痒：** 模型极度厌恶高波动。对于像 NVDA 这种疯涨疯跌的票，模型会强制你**轻仓**，防止你在震荡中被洗出去。
3.  **资金套利：** 利用 **深度实值 LEAPS**，只用 40% 的本金控制 100% 的市值。剩下的 60% 现金买美债（SGOV），利息足够覆盖期权损耗。

---

### 🚀 核心功能

* **数据说话：** 自动从 3 年历史数据中提取回归速度 ($\lambda$) 和长期波动率 ($\sigma$)，拒绝拍脑袋。
* **估值修正：** 独创 $C_{vol}$ 系数。当股价打折时，模型会自动降低风险惩罚，让你在底部敢于重仓。
* **铁壁风控：** 内置 **半凯利 (Half-Kelly)** 和 **资金硬顶**，确保无论模型多看好，你永远留有后手。

---

### ⚠️ 风险提示

本项目为量化风控研究框架。期权有风险，入市需谨慎。

---

*Designed for the Rational Investor.*
