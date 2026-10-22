# Run in one session after installing requirements.txt.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.calibration import calibration_curve

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
rng = np.random.default_rng(42)
n = 10000
past_spend = rng.uniform(0, 200, n)
activity = rng.integers(0, 11, n)
log_odds = -2 + 0.01 * past_spend + 0.2 * activity
true_ps = 1 / (1 + np.exp(-log_odds))
treatment = rng.binomial(1, true_ps)
baseline = 20 + 0.5 * past_spend + 5 * activity
true_effect = 10 + 2 * activity
noise = rng.normal(0, 10, n)
outcome = baseline + treatment * true_effect + noise
df = pd.DataFrame({'past_spend': past_spend, 'activity': activity,
                   'treatment': treatment, 'outcome': outcome})
true_sample_ate = true_effect.mean()
true_population_ate = 20.0
print(df.head().round(2))
print('Sample true effect:', round(true_sample_ate, 3), '| Population ATE:', true_population_ate)

assert not df.isna().any().any()
assert set(df['treatment'].unique()) == {0, 1}
assert np.isfinite(df.to_numpy()).all()
print(df.groupby('treatment').agg(n=('outcome', 'size'), past_spend=('past_spend', 'mean'),
      activity=('activity', 'mean'), outcome=('outcome', 'mean')).round(3))
naive_difference = df.loc[df.treatment == 1, 'outcome'].mean() - df.loc[df.treatment == 0, 'outcome'].mean()
print('Naive difference:', round(naive_difference, 3))

def cross_fit(data, ps_cols, outcome_cols, seed=42):
    T = data['treatment'].to_numpy()
    Y = data['outcome'].to_numpy()
    pred = {name: np.full(len(data), np.nan) for name in
            ['ps', 'mu0', 'mu1', 'base_ps', 'base0', 'base1']}
    counts = np.zeros(len(data), dtype=int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(data, T):
        train = data.iloc[train_idx]
        valid = data.iloc[valid_idx]
        ps_model = Pipeline([('scaler', StandardScaler()),
                             ('model', LogisticRegression(max_iter=1000))])
        ps_model.fit(train[ps_cols], train['treatment'])
        pred['ps'][valid_idx] = ps_model.predict_proba(valid[ps_cols])[:, 1]
        pred['base_ps'][valid_idx] = train['treatment'].mean()
        for state in [0, 1]:
            group = train.loc[train['treatment'] == state]
            model = LinearRegression()
            model.fit(group[outcome_cols], group['outcome'])
            pred[f'mu{state}'][valid_idx] = model.predict(valid[outcome_cols])
            pred[f'base{state}'][valid_idx] = group['outcome'].mean()
        counts[valid_idx] += 1
    assert (counts == 1).all()  # Each user receives exactly one held-out prediction
    assert all(np.isfinite(v).all() for v in pred.values())
    return pred

features = ['past_spend', 'activity']
pred = cross_fit(df, features, features)
for name in ['ps', 'mu0', 'mu1']:
    df[name] = pred[name]
print(df[['treatment', 'outcome', 'ps', 'mu0', 'mu1']].head().round(3))

T = df['treatment'].to_numpy()
Y = df['outcome'].to_numpy()
ps = df['ps'].to_numpy()
ps_metrics = pd.DataFrame([
    {'model': name, 'ROC_AUC': roc_auc_score(T, p), 'Brier': brier_score_loss(T, p),
     'Log_loss': log_loss(T, p)}
    for name, p in [('OOF PS', ps), ('OOF constant baseline', pred['base_ps'])]])
print(ps_metrics.round(4))
calibration = pd.DataFrame({'ps': ps, 'T': T})
calibration['bin'] = pd.qcut(ps, 10, duplicates='drop')
print(calibration.groupby('bin', observed=True).agg(n=('T', 'size'), predicted=('ps', 'mean'),
      observed=('T', 'mean')).round(3))
observed, predicted = calibration_curve(T, ps, n_bins=10, strategy='quantile')
plt.figure(figsize=(6, 4))
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.plot(predicted, observed, 'o-', label='OOF PS')
plt.xlabel('Mean predicted probability'); plt.ylabel('Observed treatment rate')
plt.title('PS calibration'); plt.legend(); plt.tight_layout(); plt.show()

bins = np.linspace(0, 1, 11)
ps_bins = pd.cut(ps, bins=bins, include_lowest=True)
overlap_counts = pd.crosstab(ps_bins, T).reindex(columns=[0, 1], fill_value=0)
print('Counts by PS bin:\n', overlap_counts)
print('Share with PS outside [0.05, 0.95]:', round(np.mean((ps < .05) | (ps > .95)), 4))
spend_bin = pd.cut(df.past_spend, bins=[0, 50, 100, 150, 200], include_lowest=True)
activity_bin = pd.cut(df.activity, bins=[-1, 3, 7, 10])
support_grid = pd.crosstab([spend_bin, activity_bin], df.treatment).reindex(columns=[0, 1], fill_value=0)
print('Covariate grid counts (coarse diagnostic only):\n', support_grid)
plt.figure(figsize=(7, 4))
for state, label in [(0, 'No coupon'), (1, 'Coupon')]:
    plt.hist(ps[T == state], bins=np.linspace(0, 1, 21), density=True, alpha=.5, label=label)
plt.xlabel('OOF propensity score'); plt.ylabel('Density'); plt.legend()
plt.title('Treatment / control overlap'); plt.tight_layout(); plt.show()

def weights_from_ps(t, p):
    if not np.all((p > 0) & (p < 1)):
        raise ValueError('PS contains 0 or 1: inspect support/model before estimating.')
    return np.where(t == 1, 1 / p, 1 / (1 - p))

def weight_diagnostics(t, w):
    rows = []
    for state in [0, 1]:
        a = w[t == state]
        ess = a.sum() ** 2 / np.square(a).sum()
        rows.append({'T': state, 'n': len(a), 'mean': a.mean(), 'p99': np.quantile(a, .99),
                     'max': a.max(), 'ESS': ess, 'ESS_fraction': ess / len(a),
                     'top_1pct_weight_share': np.sort(a)[-max(1, int(np.ceil(.01*len(a)))):].sum()/a.sum()})
    return pd.DataFrame(rows)

def balance_diagnostics(data, t, w):
    check = data[features].copy()
    check['past_spend_squared'] = data.past_spend ** 2
    check['activity_squared'] = data.activity ** 2
    check['spend_x_activity'] = data.past_spend * data.activity
    rows = []
    for col in check:
        a = check.loc[t == 1, col].to_numpy()
        b = check.loc[t == 0, col].to_numpy()
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        wa = np.average(a, weights=w[t == 1])
        wb = np.average(b, weights=w[t == 0])
        rows.append({'feature': col, 'treated_before': a.mean(), 'control_before': b.mean(),
                     'treated_after': wa, 'control_after': wb,
                     'SMD_before': (a.mean()-b.mean())/sd if sd else np.nan,
                     'SMD_after': (wa-wb)/sd if sd else np.nan})
    return pd.DataFrame(rows)

w = weights_from_ps(T, ps)
df['weight'] = w
weight_table = weight_diagnostics(T, w)
balance_table = balance_diagnostics(df, T, w)
print(weight_table.round(3))
print(balance_table.round(3))
plt.figure(figsize=(7, 4))
plt.plot(balance_table.SMD_before.abs(), balance_table.feature, 'o', label='Before')
plt.plot(balance_table.SMD_after.abs(), balance_table.feature, 'o', label='After')
plt.axvline(.1, ls='--', color='gray', label='0.1 reference')
plt.xlabel('Absolute SMD'); plt.title('Covariate balance'); plt.legend(); plt.tight_layout(); plt.show()

outcome_rows = []
for state in [0, 1]:
    mask = T == state
    for label, values in [('Outcome model', pred[f'mu{state}']),
                          ('Training-group mean baseline', pred[f'base{state}'])]:
        outcome_rows.append({'state': state, 'model': label, 'n': mask.sum(),
            'MAE': mean_absolute_error(Y[mask], values[mask]),
            'RMSE': np.sqrt(mean_squared_error(Y[mask], values[mask])),
            'mean_residual': (Y[mask]-values[mask]).mean()})
outcome_metrics = pd.DataFrame(outcome_rows)
print(outcome_metrics.round(3))
factual_prediction = np.where(T == 1, pred['mu1'], pred['mu0'])
residuals = df[['treatment', 'past_spend', 'activity']].copy()
residuals['residual'] = Y - factual_prediction
for col in features:
    residuals['bin'] = pd.qcut(residuals[col], 4, duplicates='drop')
    print('Residuals by', col)
    print(residuals.groupby(['treatment', 'bin'], observed=True).agg(
        n=('residual', 'size'), mean_residual=('residual', 'mean'),
        residual_sd=('residual', 'std')).round(3))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for state, ax in zip([0, 1], axes):
    mask = T == state
    ax.scatter(factual_prediction[mask], (Y-factual_prediction)[mask], s=5, alpha=.15)
    ax.axhline(0, color='black', ls='--')
    ax.set(xlabel='OOF prediction', ylabel='Actual - prediction', title=f'Factual residuals: T={state}')
plt.tight_layout(); plt.show()

def aipw_scores(y, t, p, m0, m1):
    weights_from_ps(t, p)  # Validate probabilities; do not silently clip them
    return m1-m0 + t*(y-m1)/p - (1-t)*(y-m0)/(1-p)

ipw_ate = np.average(Y[T == 1], weights=w[T == 1]) - np.average(Y[T == 0], weights=w[T == 0])
df['predicted_effect'] = df.mu1 - df.mu0
df['correction_1'] = T * (Y-df.mu1) / ps
df['correction_0'] = (1-T) * (Y-df.mu0) / (1-ps)
df['dr_score'] = aipw_scores(Y, T, ps, pred['mu0'], pred['mu1'])
dr_ate = df.dr_score.mean()
se = df.dr_score.std(ddof=1) / np.sqrt(n)
ci_low, ci_high = dr_ate - 1.96*se, dr_ate + 1.96*se
ate_table = pd.DataFrame({'method': ['Naive', 'Normalized IPW', 'Outcome only', 'AIPW'],
    'estimate': [naive_difference, ipw_ate, df.predicted_effect.mean(), dr_ate]})
ate_table['error_vs_sample_truth'] = ate_table.estimate - true_sample_ate
print(ate_table.round(3))
print(f'AIPW population-ATE approximate 95% CI: [{ci_low:.3f}, {ci_high:.3f}]; SE={se:.3f}')
print('Mean correction:', round((df.correction_1-df.correction_0).mean(), 5))

bad_outcome = cross_fit(df, features, ['past_spend'])
bad_ps = cross_fit(df, ['past_spend'], features)
both_bad = cross_fit(df, ['past_spend'], ['past_spend'])
scenarios = [('Both full', pred), ('PS full, outcome omits activity', bad_outcome),
             ('PS omits activity, outcome full', bad_ps), ('Both omit activity', both_bad)]
scenario_rows = []
for label, p in scenarios:
    value = aipw_scores(Y, T, p['ps'], p['mu0'], p['mu1']).mean()
    scenario_rows.append({'scenario': label, 'outcome_only': (p['mu1']-p['mu0']).mean(),
                          'AIPW': value, 'AIPW_error_vs_sample_truth': value-true_sample_ate})
scenario_table = pd.DataFrame(scenario_rows)
print(scenario_table.round(3))
print('Bad-PS weighted balance:')
print(balance_diagnostics(df, T, weights_from_ps(T, bad_ps['ps']))[
    ['feature', 'SMD_before', 'SMD_after']].round(3))
print('Bad-outcome factual MAE by state:')
for state in [0, 1]:
    mask = T == state
    print(state, round(mean_absolute_error(Y[mask], bad_outcome[f'mu{state}'][mask]), 3))

stress = df[['past_spend', 'activity']].copy()
stress_rng = np.random.default_rng(2026)
stress_true_ps = 1/(1+np.exp(-0.15*(stress.past_spend.to_numpy()-100)))
stress_t = stress_rng.binomial(1, stress_true_ps)
stress['treatment'] = stress_t
stress['outcome'] = baseline + stress_t*true_effect + stress_rng.normal(0, 10, n)
stress_pred = cross_fit(stress, features, features)
stress_w = weights_from_ps(stress_t, stress_pred['ps'])
print('PS outside [0.05, 0.95]:', round(np.mean((stress_pred['ps']<.05)|(stress_pred['ps']>.95)), 3))
print(weight_diagnostics(stress_t, stress_w).round(3))
print(pd.crosstab(pd.cut(stress.past_spend, [0, 50, 100, 150, 200], include_lowest=True), stress_t))
plt.figure(figsize=(7, 4))
for state in [0, 1]:
    plt.hist(stress_pred['ps'][stress_t == state], bins=np.linspace(0, 1, 21), alpha=.5,
             density=True, label=f'T={state}')
plt.title('Stress scenario: poor overlap'); plt.xlabel('OOF PS'); plt.ylabel('Density')
plt.legend(); plt.tight_layout(); plt.show()
print('Do not treat a computed full-population estimate as reliable without support.')

diagnostics = pd.Series({
    'PS Brier': ps_metrics.loc[0, 'Brier'],
    'PS baseline Brier': ps_metrics.loc[1, 'Brier'],
    'Share PS outside [0.05,0.95]': np.mean((ps<.05)|(ps>.95)),
    'Max absolute weighted SMD (checked terms)': balance_table.SMD_after.abs().max(),
    'Minimum ESS fraction': weight_table.ESS_fraction.min(),
    'Maximum weight': w.max(),
    'AIPW estimate': dr_ate,
    'AIPW approximate CI low': ci_low,
    'AIPW approximate CI high': ci_high,
})
print(diagnostics.round(4))
print('Review terms with |SMD| >= 0.1:',
      balance_table.loc[balance_table.SMD_after.abs() >= .1, 'feature'].tolist())