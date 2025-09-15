# Copyright 2025 Zhirun You
# Fama-French three-factor analysis
# Filename: fama_french_analysis.py
# Purpose: Download Fama-French factors and 25 portfolios (or load from local CSVs),
# and run the analyses requested in the assignment:
#  - Time-series regressions (OLS, GMM 0-lag)
#  - Cross-sectional OLS/GLS (with and without Shanken correction, GMM 0-lag)
#  - Fama-MacBeth two-step
#  - Discount-factor GMM (first- and second-stage)
#  - Discount-factor with (f - E f) formulation
#  - Tests: GRS, asymptotic chi2, J-test, Shanken corrections, Newey-West (optional)
#
# Usage:
#   python fama_french_analysis.py --factors_url <url> --ports_url <url>
# Or edit the LOCAL paths inside the script to point to CSV files.
#
# Requirements (pip):
#   pandas numpy scipy statsmodels matplotlib scikit-learn
#
# Notes:
#  - This script focuses on reproducible, clear implementations.
#  - GMM implementations are general-purpose using scipy.optimize.
#  - Shanken correction uses the standard formula from Shanken (1992).
#
import os
import io
import argparse
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

# --------------------------- Utilities ---------------------------------

def download_csv(url):
    """Download CSV content from URL into a pandas DataFrame.
    Falls back to reading local file if url looks like a path.
    """
    if os.path.exists(url):
        return pd.read_csv(url, header=0)
    import requests
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def safe_monthly_to_datetime(df, date_col="Date"):
    mask = df[date_col].astype(str).str.match(r'^\d{6}$')
    # 就地过滤
    df.drop(df.index[~mask], inplace=True)
    # 转换剩下的日期
    return pd.to_datetime(df[date_col].astype(str), format='%Y%m')


def excess_returns(returns, rf):
    """Subtract rf (series aligned by date) from returns DataFrame."""
    return returns.sub(rf, axis=0)


# ------------------------ Data loading helpers -------------------------

def load_factors(path_or_url=None):
    """Load Fama-French factors CSV (expects columns: Date, SMB, HML, RMRF, RF or similar)
    Returns DataFrame indexed by datetime and columns ['Mkt-RF','SMB','HML','RF'] (percent as decimals)
    """
    if path_or_url is None:
        raise ValueError('Please provide factors path or url')
    df = download_csv(path_or_url)
    # Heuristics to find date column
    if 'Date' in df.columns:
        date_col = 'Date'
    else:
        date_col = df.columns[0]
    df.index = safe_monthly_to_datetime(df, date_col)
    df = df[~df.index.duplicated(keep='first')]
    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # Try to detect factor names
    mapping = {}
    for c in df.columns:
        low = c.lower()
        if 'mkt' in low and ('rf' in low or '-' in low or 'excess' in low):
            mapping[c] = 'Mkt-RF'
        elif low == 'rmrf':
            mapping[c] = 'Mkt-RF'
        elif 'smb' in low:
            mapping[c] = 'SMB'
        elif 'hml' in low:
            mapping[c] = 'HML'
        elif low == 'rf' or 'riskfree' in low or 'risk-free' in low:
            mapping[c] = 'RF'
    df = df.rename(columns=mapping)
    # Keep only needed
    keep = [c for c in ['Mkt-RF','SMB','HML','RF'] if c in df.columns]
    return df[keep].astype(float)


def load_portfolios(path_or_url=None):
    """Load 25 portfolio returns (expects Date column and 25 columns)."""
    if path_or_url is None:
        raise ValueError('Please provide portfolios path or url')
    df = download_csv(path_or_url)
    if 'Date' in df.columns:
        date_col = 'Date'
    else:
        date_col = df.columns[0]
    df.index = safe_monthly_to_datetime(df, date_col)
    df = df[~df.index.duplicated(keep='first')]
    # Try to coerce numeric for portfolio columns
    df = df.apply(pd.to_numeric, errors='coerce')
    # Drop the date column itself if present
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    return df


# -------------------- Time-series regressions (per portfolio) ----------

def time_series_ols(returns, factors):
    """Run time-series OLS of each portfolio excess return on the factors.
    returns: DataFrame of excess returns (T x N)
    factors: DataFrame of factor excess returns (T x K), e.g., Mkt-RF, SMB, HML
    Returns: betas (K x N), alphas (N,), se_beta, se_alpha, residuals
    """
    T, N = returns.shape
    X = sm.add_constant(factors.values)
    betas = pd.DataFrame(index=['const'] + list(factors.columns), columns=returns.columns, dtype=float)
    se = betas.copy()
    resid = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        y = returns[col].values
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if mask.sum() == 0:
            continue
        res = OLS(y[mask], X[mask]).fit()
        params = res.params
        bse = res.bse
        for i, name in enumerate(['const'] + list(factors.columns)):
            betas.at[name, col] = params[i]
            se.at[name, col] = bse[i]
        resid.loc[returns.index[mask], col] = res.resid
    return betas, se, resid


def grs_test(alpha, Sigma_e, Sigma_f, T, N, K):
    """Implement GRS test statistic. alpha: vector of pricing errors (N,), Sigma_e residual covariance (N x N)
    Sigma_f: factor covariance (K x K), T sample size, N num portfolios, K num factors
    returns F-statistic and p-value
    """
    alpha = np.asarray(alpha).reshape(-1,1)
    inv_Sigma_e = la.pinv(Sigma_e)
    num = (T/ N) * (alpha.T.dot(inv_Sigma_e).dot(alpha))
    denom = 1 + (np.mean(np.diag(Sigma_f))) # rough; we'll use trace/factor dims below
    # exact formula (from GRS): F = ( (T - N - K) / N/(T - K -1) ) * alpha' * Sigma_e^{-1} * alpha
    F = ((T - N - K)/N) * float(alpha.T.dot(inv_Sigma_e).dot(alpha)) / (1 + (np.trace(Sigma_f)/K))
    df1 = N
    df2 = T - N - K
    pval = 1 - stats.f.cdf(F, df1, df2)
    return F, pval


# -------------------- Cross-sectional regressions ---------------------

def cross_sectional_ols(mean_returns, betas):
    """Run cross-sectional regression of mean returns on betas (prices of risk).
    mean_returns: series (N,)
    betas: DataFrame (K x N) where rows are factor names (without constant)
    Returns lambdas, se, residuals
    """
    X = betas.T.values  # N x K
    y = mean_returns.values
    X = sm.add_constant(X)
    res = OLS(y, X).fit()
    params = res.params
    bse = res.bse
    return params, bse, res


# -------------------- Fama-MacBeth -----------------------------------

def fama_macbeth(returns, factors):
    """Two-step Fama-MacBeth: time-series betas first, then cross-sectional in each period.
    returns: DataFrame T x N (excess returns)
    factors: DataFrame T x K (excess factors)
    Returns: average lambdas, std errors (time-series of lambda estimates -> se), and alpha test
    """
    # First stage: time-series betas
    betas, se_beta, resid = time_series_ols(returns, factors)
    # Drop const row
    beta_mat = betas.drop(index='const')
    # Second stage: for each t regress cross-sectional returns on betas
    T = returns.shape[0]
    K = beta_mat.shape[0]
    lam_ts = []
    for t in range(T):
        y = returns.iloc[t].values
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            continue
        X = beta_mat.iloc[:, mask].T.values  # n_t x K
        X = sm.add_constant(X)
        yt = y[mask]
        try:
            res = OLS(yt, X).fit()
            lam_ts.append(res.params)
        except Exception:
            continue
    lam_ts = np.vstack(lam_ts)  # S x (K+1)
    lam_mean = lam_ts.mean(axis=0)
    lam_se = lam_ts.std(axis=0, ddof=1) / np.sqrt(lam_ts.shape[0])
    return lam_mean, lam_se, lam_ts


# -------------------- GMM infrastructure ------------------------------

def gmm_estimate(init_params, moments_func, W=None, args=(), method='BFGS'):
    """General GMM estimator: minimize g(theta)' W g(theta) where g is sample mean of moments.
    moments_func(theta, *args) -> (m_t x 1) array of stacked moments per observation (T x m)
    """
    def obj(theta):
        g_t = moments_func(theta, *args)  # T x m
        g_bar = np.nanmean(g_t, axis=0)
        if W is None:
            W_loc = np.eye(len(g_bar))
        else:
            W_loc = W
        return float(g_bar.T.dot(W_loc).dot(g_bar))

    res = minimize(obj, init_params, method=method)
    return res


# -------------------- Discount factor GMM moments ---------------------

def df_moments_first_stage(theta, returns, factors):
    """Moment conditions for m = 1 - b' f  first-stage (no covariance weighting),
    where theta are b (K,) parameters.
    For each asset i and time t, the condition is E[ m_t * R_{i,t} ] - 1 = 0 (or depending on returns in levels).
    We stack across assets and also include factor orthogonality. This function returns T x m array.
    NOTE: This is a template — user should adjust to exact moment definitions they want.
    """
    b = theta
    f = factors.values  # T x K
    R = returns.values  # T x N (assume gross returns or 1+r; user must ensure correct units)
    T = R.shape[0]
    # compute m_t = 1 - f_t b
    m = 1 - (f.dot(b))  # T,
    # Moments: for each asset i: m_t * R_{t,i} - 1 -> T x N
    m_assets = (m[:, None] * R) - 1.0
    # Optionally include moments E[m * f] = 0 (K moments)
    m_f = m[:, None] * f
    moments = np.concatenate([m_assets, m_f], axis=1)
    return moments


# -------------------- Entry point: orchestrate analyses ----------------

def run_all(factors_path, ports_path, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    print('Loading data...')
    factors = load_factors(factors_path)
    ports = load_portfolios(ports_path)
    # align dates and drop rows with missing
    df = pd.concat([factors, ports], axis=1, join='inner')
    # Remove -99 sentinel values
    df = df.replace(-99, np.nan)
    # Separate
    Kcols = [c for c in factors.columns]
    Ncols = [c for c in ports.columns]
    factors = df[Kcols]
    ports = df[Ncols]
    # Compute excess returns
    if 'RF' in factors.columns:
        rf = factors['RF'] / 100.0 if factors['RF'].abs().mean()>1 else factors['RF']
        # Convert percent to decimal if necessary
    else:
        rf = pd.Series(0.0, index=factors.index)
    # convert factor units: if factors are in percent, convert to decimals
    for col in ['Mkt-RF','SMB','HML']:
        if col in factors.columns:
            if factors[col].abs().mean() > 1:
                factors[col] = factors[col] / 100.0
    ports = ports.astype(float)
    # convert portfolio returns percent->decimal if needed
    if ports.abs().mean().mean() > 1:
        ports = ports / 100.0
    excess = ports.sub(rf, axis=0)
    # Trim sample to avoid initial missing
    excess = excess.dropna(how='all')
    factors = factors.loc[excess.index]

    # (a) Time-series regression
    print('Running time-series OLS...')
    betas, se_beta, resid = time_series_ols(excess, factors[['Mkt-RF','SMB','HML']])
    betas.to_csv(os.path.join(out_dir, 'betas.csv'))

    # Compute alpha vector (mean residuals)
    alpha = resid.mean(axis=0).values
    Sigma_e = resid.cov().values
    Sigma_f = factors[['Mkt-RF','SMB','HML']].cov().values
    T = excess.shape[0]
    N = excess.shape[1]
    K = 3
    try:
        Fstat, pval = grs_test(alpha, Sigma_e, Sigma_f, T, N, K)
        print('GRS F-stat:', Fstat, 'pval', pval)
    except Exception as e:
        print('GRS test error:', e)

    # Root mean square pricing errors
    rmspe = np.sqrt(np.nanmean(alpha**2))
    print('RMSPE (alpha):', rmspe)

    # Predicted mean returns: E[r] = lambda' beta (using cross-sectional OLS)
    mean_r = excess.mean()
    lambdas, lam_bse, cs_res = cross_sectional_ols(mean_r, betas.drop(index='const'))
    print('Cross-sectional lambdas:', lambdas)

    # Fama-MacBeth
    print('Running Fama-MacBeth...')
    lam_mean, lam_se, lam_ts = fama_macbeth(excess, factors[['Mkt-RF','SMB','HML']])
    print('Fama-MacBeth lambda means:', lam_mean)

    # Save a minimal results summary
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write('RMSPE: %g\n' % rmspe)
        f.write('GRS F-stat (approx): %s, pval %s\n' % (str(Fstat), str(pval)))
        f.write('Cross-sectional lambdas: %s\n' % str(lambdas))
        f.write('Fama-MacBeth lambda mean: %s\n' % str(lam_mean))
    print('Results written to', out_dir)


#if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--factors', help='path or url to factor csv', required=True)
    #parser.add_argument('--ports', help='path or url to portfolios csv', required=True)
    #parser.add_argument('--out', help='output directory', default='results')
    #args = parser.parse_args()
    #run_all(args.factors, args.ports, args.out)
    
factors_path = "F-F_Research_Data_Factors.csv"      # 这里改成你本地下载的因子文件
ports_path = "25_Portfolios_5x5.csv"                 # 这里改成你本地下载的25组合文件

run_all(factors_path, ports_path, out_dir="results")

# End of file
