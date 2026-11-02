---
title: "A Practical Guide to Basic Causal Inference Concepts - Do Coupons Increase Customer Spending?"
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

Suppose our goal is to estimate ATT: the average effect of coupons on customers who actually received them. Several methods can help estimate this effect, including outcome regression, propensity score weighting, and matching. Here, I will use propensity score matching (PSM) as an example.

The idea of PSM is to find comparable customers who did not receive coupons and use their spending to approximate what coupon recipients would have spent without one.

### How does PSM work?

1. **Estimate propensity scores.**

   A **propensity score** is the probability of receiving treatment given a set of observed pre-treatment characteristics. It can be estimated using a classification model, such as logistic regression, with treatment assignment as the target.

   In this case, treatment is receiving a coupon. The model estimates each customer’s probability of receiving one, where coupon assignment is 1 for recipients and 0 for non-recipients.

2. **Choose the model inputs.**

   **Confounders** are variables that influence both treatment assignment and the outcome. They can create pre-existing differences between treatment and control groups, making a simple outcome comparison misleading. Including these variables in the propensity score model helps matching adjust for those differences.

   In this case, historical spending and pre-treatment engagement are confounders. Customers with higher values are more likely to receive coupons and would also tend to spend more without a coupon.

3. **Match observations and assess comparability.**

   **Propensity score matching** pairs treated observations with untreated observations that have similar propensity scores. One approach is **1:1 matching**, which selects one control observation for each matched treated observation.

   Similar propensity scores do not guarantee identical characteristics within each pair. **Covariate balance** describes how similar the distributions of pre-treatment characteristics are between the matched groups and must be checked after matching.

   In this case, each matched coupon recipient is paired with a non-recipient. Matching quality is assessed by comparing historical spending and engagement between the matched groups.

4. **Estimate the treatment effect.**

   With adequate matching quality and under the required causal assumptions, the average outcome difference between matched treated and control observations can estimate the average treatment effect among the retained treated observations. If treated observations are excluded, the estimate does not automatically apply to the entire original treated population.

   In this case, the average subsequent spending difference between matched coupon recipients and non-recipients estimates the coupon’s effect among the retained recipients.

### 4.1 Estimate propensity scores

I use logistic regression to estimate the probability of receiving a coupon from historical spending and pre-treatment engagement. Subsequent spending is not used to fit the model or choose matches. The estimated scores are used for matching; the simulation's `true_ps` is not used.

#### Code

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Estimate coupon probabilities using pre-treatment variables only
psm_data = df.copy()
features = ['past_spend', 'activity']
ps_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
ps_model.fit(psm_data[features], psm_data['coupon_assignment'])
psm_data['propensity_score'] = ps_model.predict_proba(psm_data[features])[:, 1]

display(psm_data)
```

#### Output

| | past_spend | activity | coupon_assignment | subsequent_spending | propensity_score |
|---|---:|---:|---:|---:|---:|
| 0 | 154.791210 | 0 | 0 | 87.184820 | 0.392204 |
| 1 | 87.775688 | 7 | 1 | 116.847515 | 0.564984 |
| 2 | 171.719584 | 2 | 1 | 135.199143 | 0.531588 |
| 3 | 139.473606 | 7 | 0 | 124.498443 | 0.686560 |
| 4 | 18.835470 | 6 | 0 | 54.645706 | 0.346991 |
| ... | ... | ... | ... | ... | ... |
| 9995 | 89.349407 | 1 | 0 | 71.228992 | 0.288442 |
| 9996 | 61.869468 | 10 | 1 | 139.804893 | 0.643284 |
| 9997 | 33.069394 | 3 | 0 | 52.639808 | 0.253784 |
| 9998 | 74.682080 | 8 | 1 | 143.106050 | 0.580725 |
| 9999 | 96.181555 | 9 | 1 | 141.222840 | 0.676958 |

10000 rows × 5 columns

### 4.2 Check overlap

Before matching, I count coupon recipients and non-recipients in five propensity score ranges. This is a coarse support diagnostic, not proof that every customer has a comparable counterpart.

```python
# Count recipients and non-recipients in each propensity score range
score_bins = pd.cut(
    psm_data['propensity_score'],
    bins=np.linspace(0, 1, 6),
    include_lowest=True
)
overlap = pd.crosstab(
    score_bins,
    psm_data['coupon_assignment']
)
display(overlap)
```

#### Output

| propensity_score | coupon_assignment = 0 | coupon_assignment = 1 |
|---|---:|---:|
| (-0.001, 0.2] | 475 | 102 |
| (0.2, 0.4] | 1827 | 827 |
| (0.4, 0.6] | 1824 | 1777 |
| (0.6, 0.8] | 832 | 1808 |
| (0.8, 1.0] | 58 | 470 |

**Interpretation.** Both groups appear in every checked range, but the highest-score range has only **58 non-recipients versus 470 recipients**. Even with overall overlap, some recipients may not have a sufficiently close, unused control under 1:1 matching without replacement.

### 4.3 Perform 1:1 matching

For each coupon recipient, I select the nearest available non-recipient on the **logit propensity score** scale. I use a caliper of **0.2 standard deviations of the estimated logit scores**, approximately **0.1693** in this run. This limits the permitted distance; it is an illustrative matching choice rather than a universal optimal setting.

Matching is **without replacement**: each control is used at most once. Recipients are processed in a fixed random order (`random_state=42`). Recipients without an available control within the caliper remain unmatched. Matching order, caliper, and replacement choices can change the retained sample.

```python
# The decision function gives the logit of the estimated probability
psm_data['ps_logit'] = ps_model.decision_function(psm_data[features])
caliper = 0.2 * psm_data['ps_logit'].std(ddof=1)

# 2. Match each recipient to the nearest available non-recipient
# Use a fixed random order; each control is used at most once
recipients = psm_data.loc[psm_data.coupon_assignment == 1]
non_recipients = psm_data.loc[psm_data.coupon_assignment == 0]
available_controls = non_recipients.index.tolist()
pairs = []

for recipient_id in recipients.sample(frac=1, random_state=42).index:
    if not available_controls:
        break
    distances = (
        psm_data.loc[available_controls, 'ps_logit']
        - psm_data.loc[recipient_id, 'ps_logit']
    ).abs()
    control_id = distances.idxmin()
    if distances.loc[control_id] <= caliper:
        pairs.append((recipient_id, control_id))
        available_controls.remove(control_id)

pairs = pd.DataFrame(pairs, columns=['recipient_id', 'control_id'])
if pairs.empty:
    raise ValueError('No matches found. Review overlap and matching settings.')

matched_recipients = psm_data.loc[pairs.recipient_id]
matched_controls = psm_data.loc[pairs.control_id]

# 3. Check retention and balance before looking at the outcome difference
retention = pd.DataFrame({
    'Group': ['Coupon recipients', 'Non-recipients'],
    'Original': [len(recipients), len(non_recipients)],
    'Matched': [len(pairs), len(pairs)]
})
retention['Unmatched'] = retention['Original'] - retention['Matched']
display(retention)
```

#### Output

| | Group | Original | Matched | Unmatched |
|---|---|---:|---:|---:|
| 0 | Coupon recipients | 4984 | 3500 | 1484 |
| 1 | Non-recipients | 5016 | 3500 | 1516 |

**Interpretation.** The procedure retains **3,500 pairs**, including approximately **70.2% of the original coupon recipients**. It excludes **1,484 recipients**. An unmatched recipient does not necessarily have no similar non-recipient in the original data; a suitable control may already have been used.

Because some recipients are excluded, the estimate now targets the recipients retained in the matched sample. It should not automatically be generalized to all coupon recipients or all 10,000 customers.

### 4.4 Check covariate balance

Matching must be evaluated using the covariates themselves, not just similarity in propensity scores. Here I compare matched means and standardized mean differences (SMDs) for `past_spend` and `activity`.

For each variable, I use the original pooled standard deviation as a fixed denominator for both the before- and after-matching SMDs. Values closer to zero indicate closer means.

```python
balance_rows = []
for feature in features:
    # Keep the original pooled SD fixed for before/after comparisons
    pooled_sd = np.sqrt(
        (recipients[feature].var(ddof=1)
         + non_recipients[feature].var(ddof=1)) / 2
    )
    balance_rows.append({
        'Feature': feature,
        'Recipient mean (matched)': matched_recipients[feature].mean(),
        'Control mean (matched)': matched_controls[feature].mean(),
        'SMD before': (
            recipients[feature].mean() - non_recipients[feature].mean()
        ) / pooled_sd,
        'SMD after': (
            matched_recipients[feature].mean()
            - matched_controls[feature].mean()
        ) / pooled_sd
    })
balance = pd.DataFrame(balance_rows)
display(balance.round(3))
```

#### Output

| | Feature | Recipient mean (matched) | Control mean (matched) | SMD before | SMD after |
|---|---|---:|---:|---:|---:|
| 0 | past_spend | 99.626 | 98.840 | 0.524 | 0.014 |
| 1 | activity | 5.025 | 5.029 | 0.565 | -0.001 |

**Interpretation.** The absolute SMDs fall from approximately **0.524 to 0.014** for historical spending and **0.565 to 0.001** for engagement. These observed means are much closer after matching.

This table checks mean balance only. A more complete assessment would also inspect distributions, variances, and relevant nonlinear terms or interactions. A small SMD is not proof that all confounding has been removed. If covariate balance remains poor, the matching design should be revised before interpreting the outcome comparison.

### 4.5 Estimate the effect and compare the appropriate benchmark

For each matched pair, I subtract the non-recipient's spending from the recipient's spending, then average the differences. I convert the outcome columns to arrays so pandas does not align the different customer IDs.

Because this is simulated data, I can also calculate the true average coupon effect for the **same retained recipients**. True effects are used only for evaluation, never for deciding the matching rules or choosing pairs.

```python
# 4. Estimate the average effect among retained recipients
# Use arrays so pandas does not align different customer IDs
psm_effect = (
    matched_recipients['subsequent_spending'].to_numpy()
    - matched_controls['subsequent_spending'].to_numpy()
).mean()

# Simulation benchmark only: never use true effects to choose matches
true_effect_by_customer = pd.Series(true_effect, index=df.index)
matched_true_effect = true_effect_by_customer.loc[pairs.recipient_id].mean()

psm_comparison = pd.DataFrame({
    'Metric': [
        'Naive Difference',
        'True ATT: all coupon recipients',
        'True effect: matched coupon recipients',
        'PSM estimate: matched coupon recipients'
    ],
    'Value': [
        naive_difference,
        true_effect_by_customer.loc[recipients.index].mean(),
        matched_true_effect,
        psm_effect
    ]
})
display(psm_comparison.round(2))
```

#### Output

| | Metric | Value |
|---|---|---:|
| 0 | Naive Difference | 45.19 |
| 1 | True ATT: all coupon recipients | 21.79 |
| 2 | True effect: matched coupon recipients | 20.05 |
| 3 | PSM estimate: matched coupon recipients | 20.84 |

**Interpretation.** PSM estimates an average spending increase of approximately **$20.84 among matched coupon recipients**. The corresponding true average effect is **$20.05**. Matching substantially reduces the discrepancy seen in the original naive difference of **$45.19**, but the point estimate still differs from its simulation benchmark by approximately **$0.79**.

The true ATT for all original coupon recipients is **$21.79**. Its difference from the matched recipients' true effect reflects a change in the target population. Therefore, the PSM estimate should be compared with **$20.05**, not treated as an estimate of the full-population ATE or automatically as an estimate of ATT for every original recipient.

This section reports a point estimate and descriptive diagnostics, not a confidence interval. A causal interpretation requires sufficient overlap for the retained population, consistency and no interference, and no unmeasured confounding given the selected pre-treatment variables. These conditions follow from the way this simulation was constructed; in a real study, they would require substantive justification.

### References for matching and diagnostics

- [MatchIt: Assessing Balance](https://kosukeimai.github.io/MatchIt/articles/assessing-balance.html)
- [MatchIt: Matching Methods](https://kosukeimai.github.io/MatchIt/articles/matching-methods.html)
- [MatchIt: Matching for Causal Inference](https://kosukeimai.github.io/MatchIt/reference/matchit.html)
