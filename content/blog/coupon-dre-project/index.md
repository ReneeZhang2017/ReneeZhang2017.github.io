---
title: "Do Coupons Increase Customer Spending? A Practical Guide to Causal Inference"
date: 2026-10-21
draft: false
categories: ["Causal Inference"]
description: "A step-by-step causal inference project using simulated coupon data to estimate treatment effects."
---

> **Project type: Simulation-based learning project.** In this case study, I walk through a simplified causal inference project to demonstrate how doubly robust estimation can estimate the average effect of issuing coupons on customer spending. All results are based on simulated data and do not represent actual business outcomes at any company. 

### Jump to a step

1. [Business Context](#1-Business-Context)
2. [Generate the simulated data](#2-generate-the-simulated-data)
3. [True ATE vs. True ATT vs. Naive Difference](#3-true-ate-vs-true-att-vs-naive-difference)
4. [Adjust for Selection Bias with Propensity Score Matching](#4-adjust-for-selection-bias-with-propensity-score-matching)


## 1. Business Context
Consider an e-commerce platform that has already distributed coupons to some customers. Customers with higher historical spending and engagement were more likely to receive them. We want to know what is the average increase in customer spending caused by receiving a coupon.

I will use a simulated simple dataset of 10,000 users containing 4 variables: 
(1) historical spending
(2) pre-treatment engagement
(3) coupon assignment 
(4) subsequent spending

## 2. Generate the simulated data

The DataFrame `df` is a simulated dataset of 10,000 customers. Each row represents one customer with four observed variables:

1. **Historical spending:** I randomly generate a value between 0 and 200 for each customer using a uniform distribution. This represents how much the customer spent before coupon assignment.

2. **Pre-treatment engagement:** I randomly generate an integer from 0 to 10, inclusive, for each customer. Higher values represent greater engagement before coupon assignment.

3. **Coupon assignment:** I first calculate each customer's probability of receiving a coupon based on historical spending and engagement. Customers with higher spending and engagement have higher probabilities. I then use these probabilities to randomly assign coupons: `1` means the customer receives a coupon, and `0` means they do not. In this simulated sample, **4,984 customers (49.84%) receive a coupon**, while **5,016 (50.16%) do not**.

4. **Subsequent spending:** I first calculate expected spending without a coupon as `20 + 0.5 * past_spend + 5 * activity`. For customers who receive a coupon, I add a coupon effect of `10 + 2 * activity`, so more engaged customers receive a larger spending lift. Finally, I add normally distributed random noise with a mean of 0 and a standard deviation of 10 to every customer's spending. The final formula is `outcome = baseline + coupon_assignment * true_effect + noise`.

### Code

```python
import numpy as np
import pandas as pd

# Create a random number generator and store it in the variable 'rng'.
rng = np.random.default_rng(42)

# Set n as the number of customers, and n=10,000
n = 10000

# (1) Historical Spending: a random number between 0 to 200
past_spend = rng.uniform(0, 200, n)

# (2) pre-treatment engagement: a random integer between 0 to 10
activity = rng.integers(0, 11, n)

# (3) Coupon Assignment: customers with high past_spend and high activity are more likely to receive a coupon
# Create a coupon issuance probability to generate simulated coupon assignments.
# treatment=1 means the customer received a coupon; treatment=0 means they did not.
log_odds = -2 + 0.01 * past_spend + 0.2 * activity
true_ps = 1 / (1 + np.exp(-log_odds))
treatment = rng.binomial(1, true_ps)

# (4) Subsequent Spending
# Spending when there is no coupon assigned
baseline = 20 + 0.5 * past_spend + 5 * activity
# The true coupon effect when a customer receives a coupon
true_effect = 10 + 2 * activity
# Add noise to spending
noise = rng.normal(0, 10, n)
# The final subsequent spending = baseline + treatment * true_effect + noise
outcome = baseline + treatment * true_effect + noise

# Aggregate all data into a dataframe
df = pd.DataFrame({
    'past_spend': past_spend,
    'activity': activity,
    'coupon_assignment': treatment,
    'subsequent_spending': outcome
})

display(df.head(10))
display(df.describe())
```

### Output

{{< notebook-outputs >}}
`display(df.head(10))`

| | past_spend | activity | coupon_assignment | subsequent_spending |
|---|---:|---:|---:|---:|
| 0 | 154.791210 | 0 | 0 | 87.184820 |
| 1 | 87.775688 | 7 | 1 | 116.847515 |
| 2 | 171.719584 | 2 | 1 | 135.199143 |
| 3 | 139.473606 | 7 | 0 | 124.498443 |
| 4 | 18.835470 | 6 | 0 | 54.645706 |
| 5 | 195.124470 | 2 | 1 | 132.531542 |
| 6 | 152.227940 | 5 | 1 | 130.981857 |
| 7 | 157.212861 | 0 | 1 | 107.288876 |
| 8 | 25.622727 | 7 | 0 | 88.069488 |
| 9 | 90.077188 | 3 | 0 | 85.696460 |

<!-- output-split -->

`display(df.describe())`

| | past_spend | activity | coupon_assignment | subsequent_spending |
|---|---:|---:|---:|---:|
| count | 10000.000000 | 10000.000000 | 10000.000000 | 10000.000000 |
| mean | 99.424320 | 5.032600 | 0.498400 | 105.848177 |
| std | 57.644184 | 3.155144 | 0.500022 | 40.495370 |
| min | 0.062884 | 0.000000 | 0.000000 | -1.792771 |
| 25% | 49.207598 | 2.000000 | 0.000000 | 75.279554 |
| 50% | 99.660298 | 5.000000 | 0.000000 | 105.355246 |
| 75% | 149.135540 | 8.000000 | 1.000000 | 134.962128 |
| max | 199.995404 | 10.000000 | 1.000000 | 218.490703 |
{{< /notebook-outputs >}}



## 3. True ATE vs. True ATT vs. Naive Difference

Because this dataset is simulated, we know each customer’s true coupon effect. We can therefore calculate two benchmarks: the true ATE, which averages these effects across all 10,000 customers, and the true ATT, which averages them only among customers who actually received a coupon.

I compare both benchmarks with the naive difference: the difference in average observed spending between coupon recipients and non-recipients. This comparison illustrates why the target population matters and why a simple comparison between the two groups may not recover the causal effect.

### Code

```python
# True ATE: the average coupon effect across all 10,000 customers
# true_effect = 10 + 2 * activity
true_ate = true_effect.mean()

# True ATT: the average coupon effect among actual coupon recipients
received_coupon = df['coupon_assignment'].to_numpy() == 1
true_att = true_effect[received_coupon].mean()

# Naive difference: average observed spending with vs. without a coupon
group_mean = df.groupby('coupon_assignment')['subsequent_spending'].mean()
naive_difference = group_mean.loc[1] - group_mean.loc[0]

# Summarize the three comparisons in a DataFrame
comparison = pd.DataFrame({
    'Metric': ['True ATE', 'True ATT', 'Naive Difference'],
    'Value': [true_ate, true_att, naive_difference],
    'Interpretation': [
        'Average causal effect across all 10,000 customers: '
        'spending if everyone received a coupon versus if no one did.',

        'Average causal effect among actual coupon recipients: '
        'their spending with a coupon versus without one.',

        'Difference in average observed spending between '
        'coupon recipients and non-recipients.'
    ]
})

display(
    comparison.style
    .format({'Value': '{:.2f}'})
)

# Compare pre-treatment characteristics and subsequent spending by group
display(
    df.groupby('coupon_assignment')[['past_spend', 'activity', 'subsequent_spending']].mean()
)
```

### Output

| | Metric | Value | Interpretation |
|---|---|---:|---|
| 0 | True ATE | 20.07 | Average causal effect across all 10,000 customers: spending if everyone received a coupon versus if no one did. |
| 1 | True ATT | 21.79 | Average causal effect among actual coupon recipients: their spending with a coupon versus without one. |
| 2 | Naive Difference | 45.19 | Difference in average observed spending between coupon recipients and non-recipients. |

| coupon_assignment | past_spend | activity | subsequent_spending |
|---|---:|---:|---:|
| 0 | 84.869642 | 4.178030 | 83.325577 |
| 1 | 114.072446 | 5.892657 | 128.515383 |

**Interpretation.**

**1. True ATE vs. True ATT: effects for different target populations.**

ATE (Average Treatment Effect) averages the causal effect across all 10,000 customers. ATT (Average Treatment Effect on the Treated) averages it only among the 4,984 customers who actually received coupons. Each compares its target customers under two scenarios: receiving a coupon and not receiving one. In real-world data, we cannot observe both outcomes for the same customer under otherwise identical conditions. Here, we know both benchmarks because we specified the simulation’s effect mechanism.

The true sample ATE is **$20.07**, while the true sample ATT is **$21.79**. Coupon recipients have higher average engagement, and we defined the coupon effect as `10 + 2 * activity`. Their average coupon effect is therefore larger. This difference reflects the target populations and varying coupon effects; it is not an estimation error.

**2. Naive Difference vs. both causal benchmarks: selection bias.**

The naive difference simply subtracts the average observed spending of non-recipients from that of coupon recipients, without adjusting for differences between the groups.

The naive difference is **$45.19** ($128.52 − $83.33), substantially higher than both the true ATE of **$20.07** and the true ATT of **$21.79**.

Coupon recipients already had higher historical spending (**114.07 vs. 84.87**) and engagement (**5.89 vs. 4.18**). These characteristics also increase spending without a coupon. This creates selection bias: customers who would tend to spend more anyway are more likely to receive a coupon.

As a result, the naive difference mixes the coupon’s causal effect with pre-existing differences between the groups. It overstates both ATE and ATT in this simulation.

## 4. Adjust for Selection Bias with Propensity Score Matching

The naive difference compares groups with different historical spending and engagement. To make a more comparable comparison, I use propensity score matching (PSM).

I first estimate each customer’s probability of receiving a coupon using historical spending and pre-treatment engagement. I then match coupon recipients with non-recipients who have similar estimated probabilities.

The initial target is ATT: the average coupon effect among actual coupon recipients. However, if some recipients cannot be matched, the estimate applies to the recipients retained in the matched sample.

