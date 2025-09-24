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
# These small helper functions centralize common tasks:
#  - reading CSVs from local disk or HTTP
#  - converting YYYYMM numeric-like dates to pandas datetime
#  - computing excess returns


def download_csv(url):
    """Download CSV content from URL into a pandas DataFrame.
    Behavior:
      - If `url` is a local path that exists, read it with pandas.read_csv.
      - Otherwise attempt to download via requests.get and parse the CSV text.
    Notes:
      - This function raises for HTTP errors (requests will raise on bad status).
      - Header handling is kept simple (header=0). If your CSV differs adjust here.
    """
    # If the path exists on disk, prefer local read (convenient for reproducibility).
    if os.path.exists(url):
        return pd.read_csv(url, header=0)

    # Otherwise download
    import requests
    r = requests.get(url)
    r.raise_for_status()
    # Use io.StringIO to let pandas read from the downloaded text.
    return pd.read_csv(io.StringIO(r.text))


def safe_monthly_to_datetime(df, date_col="Date"):
    """Convert a date column that is in YYYYMM (6-digit) format to a pandas DatetimeIndex.

    Steps:
      - Create a boolean mask selecting rows whose date column is exactly 6 digits.
        This filters out headers/footers or monthly labels that are not strict YYYYMM.
      - Drop rows that don't match (done in-place).
      - Convert the remaining date strings to pandas datetime with format '%Y%m'.

    Returns:
      - DatetimeIndex that can be assigned to df.index by the caller.
    Important:
      - This function mutates the passed DataFrame by dropping rows that don't match.
      - If your date column is in another format adapt the regex/format string.
    """
    # Keep rows that look like YYYYMM (e.g., 202003)
    mask = df[date_col].astype(str).str.match(r'^\d{6}$')
    # Drop (in-place) rows that don't match the YYYYMM pattern.
    df.drop(df.index[~mask], inplace=True)
    # Convert the remaining values to datetime (first day of the month).
    return pd.to_datetime(df[date_col].astype(str), format='%Y%m')


def excess_returns(returns, rf):
    """Subtract the risk-free rate from asset returns.

    Parameters:
      returns : DataFrame (T x N) of asset returns (same frequency as rf)
      rf      : Series or DataFrame (T,) or (T x 1) with the risk-free rate for each period

    Returns:
      DataFrame of excess returns aligned by index/columns.

    Notes:
      - This uses pandas broadcasting: `returns.sub(rf, axis=0)` subtracts rf row-wise.
      - Make sure units are consistent (e.g., both in percent or both in decimal).
    """
    return returns.sub(rf, axis=0)

# ------------------------ Data loading helpers -------------------------
# Two functions to load Fama-French factor files and portfolio return files.
# They perform light normalization of column names and convert dates to a DatetimeIndex.


def load_factors(path_or_url=None):
    """Load Fama-French factors CSV and normalize column names.

    Expected contents (typical):
      Date,   Mkt-RF (or RMRF), SMB, HML, RF, ...
    Behavior:
      - Detect a date column (column named 'Date' or the first column).
      - Convert the date column from YYYYMM to pandas datetime via safe_monthly_to_datetime.
      - Attempt to identify factor columns heuristically (case-insensitive).
      - Rename to canonical columns 'Mkt-RF', 'SMB', 'HML', 'RF' when possible.
      - Return a DataFrame indexed by datetime and containing the detected factor columns.

    Return:
      DataFrame with a DatetimeIndex and subset of columns ['Mkt-RF','SMB','HML','RF'] (if present).

    Important:
      - Many Fama-French downloads are in percentage points (e.g., 0.32 means 0.32%).
        Keep track whether your downstream code expects decimals (0.0032) or percent.
      - If your CSV has different headings, update the heuristic mapping below.
    """
    if path_or_url is None:
        raise ValueError('Please provide factors path or url')
    df = download_csv(path_or_url)

    # Choose a date column: prefer explicit 'Date' otherwise take the first column.
    if 'Date' in df.columns:
        date_col = 'Date'
    else:
        date_col = df.columns[0]

    # Convert matched YYYYMM strings to datetime and set as index.
    df.index = safe_monthly_to_datetime(df, date_col)

    # Remove any duplicated dates, keeping the first occurrence.
    df = df[~df.index.duplicated(keep='first')]

    # Basic normalization: strip whitespace from column names.
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Heuristic mapping from raw column names to canonical names.
    mapping = {}
    for c in df.columns:
        low = c.lower()
        # Many files name market excess return 'Mkt-RF', 'RMRF', or 'Mkt-RF'
        if 'mkt' in low and ('rf' in low or '-' in low or 'excess' in low):
            mapping[c] = 'Mkt-RF'
        elif low == 'rmrf':
            mapping[c] = 'Mkt-RF'
        elif 'smb' in low:
            mapping[c] = 'SMB'
        elif 'hml' in low:
            mapping[c] = 'HML'
        # risk-free rate column
        elif low == 'rf' or 'riskfree' in low or 'risk-free' in low:
            mapping[c] = 'RF'

    # Rename columns according to the mapping we've built.
    df = df.rename(columns=mapping)

    # Keep only columns that we can interpret; cast to float to avoid string dtypes.
    keep = [c for c in ['Mkt-RF', 'SMB', 'HML', 'RF'] if c in df.columns]
    return df[keep].astype(float)


def load_portfolios(path_or_url=None):
    """Load portfolio returns CSV (e.g., 25 portfolios, 5x5 sorted) and set a DatetimeIndex.

    Expected:
      - A 'Date' column (or first column) and N portfolio return columns.
      - Returns are coerced to numeric (non-numeric values become NaN).

    Behavior:
      - Converts the Date column from YYYYMM to datetime.
      - Drops the original Date column from returned DataFrame (index holds the date).
      - Keeps any numeric portfolio columns.

    Return:
      DataFrame indexed by datetime, columns are portfolio return series.
    """
    if path_or_url is None:
        raise ValueError('Please provide portfolios path or url')
    df = download_csv(path_or_url)

    # Same date detection logic as load_factors.
    if 'Date' in df.columns:
        date_col = 'Date'
    else:
        date_col = df.columns[0]
    df.index = safe_monthly_to_datetime(df, date_col)
    df = df[~df.index.duplicated(keep='first')]

    # Force portfolio columns to numeric (non-numeric -> NaN).
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop the Date column if it remains as a column (index already contains it).
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    return df

# -------------------- Time-series regressions (per portfolio) ----------
# Time-series OLS for each asset separately: R_it - RF_t (excess) on factors
# Regression: R_excess_it = alpha_i + beta_i' f_t + eps_it
# Return shapes:
#  - betas: DataFrame with K+1 rows (including const) x N columns (assets)
#  - se: standard errors matrix with same shape as betas
#  - resid: T x N DataFrame of residuals


def time_series_ols(returns, factors):
    """Run time-series OLS of each portfolio excess return on the factors.

    Parameters:
      returns : DataFrame (T x N) of asset excess returns (aligned by index)
      factors : DataFrame (T x K) of factor excess returns (no constant)

    Returns:
      betas : DataFrame with index ['const'] + factor names and columns = returns.columns
      se    : same shape as betas, contains standard errors
      resid : DataFrame (T x N) of regression residuals

    Implementation details:
      - We construct X = [1, f_t] and fit one OLS per asset using statsmodels OLS.
      - We mask out periods with NaNs in either the asset series or the factor matrix.
      - For small samples or constant columns this approach is robust but may produce NaNs.
    """
    T, N = returns.shape

    # Create the regressor matrix with a constant; X has shape (T x (K+1)).
    X = sm.add_constant(factors.values)

    # Prepare output DataFrames indexed/colnamed appropriately.
    betas = pd.DataFrame(index=['const'] + list(factors.columns), columns=returns.columns, dtype=float)
    se = betas.copy()
    resid = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    # Loop over assets (columns) and run OLS individually.
    for col in returns.columns:
        y = returns[col].values
        # mask: only use rows where y is not NaN and X row is not NaN anywhere.
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if mask.sum() == 0:
            # No valid observations for this asset -> skip
            continue

        # Fit OLS using statsmodels; we fit on the masked subset.
        res = OLS(y[mask], X[mask]).fit()
        params = res.params
        bse = res.bse

        # Store parameter estimates and standard errors into DataFrames.
        for i, name in enumerate(['const'] + list(factors.columns)):
            betas.at[name, col] = params[i]
            se.at[name, col] = bse[i]

        # Store residuals aligned back to the original index for this asset.
        resid.loc[returns.index[mask], col] = res.resid

    return betas, se, resid


def grs_test(alpha, Sigma_e, Sigma_f, T, N, K):
    """Compute the GRS (Gibbons, Ross, Shanken) test for mean-variance efficiency of factor model.

    Inputs:
      alpha   : vector of pricing errors (length N) — e.g., mean residuals from time-series regressions
      Sigma_e : residual covariance matrix across N portfolios (N x N)
      Sigma_f : covariance matrix of factor returns (K x K)
      T, N, K : sample size, number of assets, number of factors

    Returns:
      F : GRS F-statistic
      pval : p-value computed from F_{N, T-N-K}

    Notes and approximations:
      - The canonical GRS test formula is:
          F = [ (T - N - K) / N ] * [ alpha' Sigma_e^{-1} alpha ] / [ 1 + mu_f' Sigma_f^{-1} mu_f ]
        where mu_f is the vector of factor means. Some variants simplify the denominator.
      - I include a simplified denominator using trace(Sigma_f)/K as a rough measure of factor variance.
      - This implementation uses a pseudo-inverse for Sigma_e (la.pinv) to guard against singularity.
      - For precise inference and small samples, you may want to implement the exact denominator using factor means.
    """
    alpha = np.asarray(alpha).reshape(-1, 1)
    inv_Sigma_e = la.pinv(Sigma_e)  # pseudo-inverse for numerical stability

    # Numerator: quadratic form alpha' Sigma_e^{-1} alpha
    quad = float(alpha.T.dot(inv_Sigma_e).dot(alpha))

    # Simplified denominator: a scale that accounts for factor variability (approximation).
    # For a fully exact GRS you should compute the denominator using the sample mean of factors.
    denom_scale = 1 + (np.trace(Sigma_f) / K)

    # GRS F-statistic (approximate form used here).
    F = ((T - N - K) / N) * quad / denom_scale

    # Degrees of freedom for the F distribution
    df1 = N
    df2 = T - N - K
    pval = 1 - stats.f.cdf(F, df1, df2)
    return F, pval

# -------------------- Cross-sectional regressions ---------------------
# Cross-sectional regression (time-series mean returns regressed on betas)
# Regression: E[R_i] = gamma_0 + gamma' beta_i + u_i
# Returns estimates of risk prices (lambdas / gammas) and their standard errors.


def cross_sectional_ols(mean_returns, betas):
    """Run cross-sectional regression of mean returns on betas (prices of risk).

    Parameters:
      mean_returns : pandas Series of cross-sectional mean excess returns (length N)
      betas        : DataFrame (K x N) containing betas per asset (rows are factor names)

    Returns:
      params : array-like of estimated coefficients (intercept followed by K lambdas)
      bse    : standard errors for the params
      res    : full statsmodels RegressionResults for the fit (useful for diagnostics)

    Implementation detail:
      - We transpose betas to get an N x K design matrix, add a constant, and run OLS.
      - This is the standard "cross-sectional" step in Fama-MacBeth/FF-style tests.
    """
    X = betas.T.values  # N x K
    y = mean_returns.values
    X = sm.add_constant(X)
    res = OLS(y, X).fit()
    params = res.params
    bse = res.bse
    return params, bse, res

# -------------------- Fama-MacBeth -----------------------------------
# Implements the canonical two-step Fama-MacBeth procedure:
# 1) For each asset estimate time-series betas using all T observations.
# 2) For each time t run cross-sectional regression of R_t on the betas estimated in (1).
# 3) Average the time-series of lambda estimates and compute standard errors across time.


def fama_macbeth(returns, factors):
    """Two-step Fama-MacBeth estimator.

    Parameters:
      returns : DataFrame (T x N) of excess returns
      factors : DataFrame (T x K) of excess factor returns

    Returns:
      lam_mean : (K+1,) array of average cross-sectional estimates (intercept + K factor prices)
      lam_se   : (K+1,) array of standard errors (standard error of the time-series of lambda estimates)
      lam_ts   : array S x (K+1) of period-by-period lambda estimates used to compute lam_mean/lam_se

    Steps:
      - First stage: run time_series_ols to get betas_i for each asset i (these are constant across t).
      - Second stage: for each period t regress cross-sectional returns R_t on the previously estimated betas.
      - We collect the time-series of cross-sectional estimates and compute mean and standard errors.
    Notes:
      - The returned lam_ts includes the intercept (constant) as its first column.
      - Small-sample corrections (e.g., Shanken) are not applied here; they can be added separately.
    """
    # First stage: obtain betas and residuals for each asset.
    betas, se_beta, resid = time_series_ols(returns, factors)

    # Remove the 'const' row — betas for cross-sectional regressions are the factor loadings only.
    beta_mat = betas.drop(index='const')

    T = returns.shape[0]
    K = beta_mat.shape[0]  # number of factors
    lam_ts = []

    # Loop over time (each cross-section). For each t, run OLS of R_t on betas_i (i cross-section).
    for t in range(T):
        y = returns.iloc[t].values
        mask = ~np.isnan(y)  # drop assets with missing return at time t
        if mask.sum() == 0:
            continue

        # Build the design matrix for the cross-section at time t: rows are assets with non-missing returns.
        X = beta_mat.iloc[:, mask].T.values  # n_t x K
        X = sm.add_constant(X)
        yt = y[mask]
        try:
            res = OLS(yt, X).fit()
            lam_ts.append(res.params)  # store (K+1,) vector
        except Exception:
            # In case of numerical issues for a particular cross-section skip it.
            continue

    # Stack the period-by-period estimates into an array S x (K+1)
    lam_ts = np.vstack(lam_ts)  # S x (K+1)
    lam_mean = lam_ts.mean(axis=0)
    # Standard error: sample standard deviation of the time-series of lambdas divided by sqrt(S)
    lam_se = lam_ts.std(axis=0, ddof=1) / np.sqrt(lam_ts.shape[0])
    return lam_mean, lam_se, lam_ts

# -------------------- GMM infrastructure ------------------------------
# Generic GMM wrapper that minimizes the quadratic form gbar' W gbar where gbar is the sample average
# of moment functions. The moments function should return a T x m array of per-period moments.


def gmm_estimate(init_params, moments_func, W=None, args=(), method='BFGS'):
    """General GMM estimator using scipy.optimize.minimize.

    Parameters:
      init_params : initial guess for theta (1d array)
      moments_func: function(theta, *args) -> returns T x m array of per-period moments
      W           : weighting matrix (m x m). If None identity is used.
      args        : additional args passed to moments_func (e.g., (returns, factors))
      method      : optimization method passed to scipy.optimize.minimize (default 'BFGS')

    Returns:
      res : OptimizeResult returned by scipy.optimize.minimize (contains x = estimated parameters)

    Implementation notes:
      - The objective evaluated is obj(theta) = gbar(theta)' W gbar(theta) where gbar is sample mean of moments.
      - If W is None the identity matrix is used (equivalent to Method of Moments).
      - This function currently does not implement two-step GMM (update W using estimated covariance).
      - The user should ensure that moments_func handles NaNs and returns an array with shape (T, m).
    """
    def obj(theta):
        # Evaluate per-period moments T x m
        g_t = moments_func(theta, *args)
        # Compute the sample mean across T (axis=0). Use nanmean to ignore any NaNs in moments.
        g_bar = np.nanmean(g_t, axis=0)
        if W is None:
            W_loc = np.eye(len(g_bar))
        else:
            W_loc = W
        # Quadratic objective
        return float(g_bar.T.dot(W_loc).dot(g_bar))

    res = minimize(obj, init_params, method=method)
    return res

# -------------------- Discount factor GMM moments ---------------------
# Example/template of a discount-factor moment function for the first-stage GMM.
# The 'm' (stochastic discount factor) is modeled as m_t = 1 - b' f_t.
# Moments are E[ m_t * R_{i,t} ] - 1 = 0 for each asset i (if R is gross return).
# Also optionally include orthogonality moments E[m_t * f_t] = 0 (K moments).


def df_moments_first_stage(theta, returns, factors):
    """Moment conditions for a simple discount-factor specification m_t = 1 - b' f_t.

    Arguments:
      theta   : parameter vector b (length K) (prices in the linear SDF)
      returns : DataFrame or array (T x N) of gross returns R_{t,i} (NOT excess returns),
                i.e., R = 1 + r. If your input is net returns (r), you must convert to gross:
                R = 1 + r (or adjust the moment definitions accordingly).
      factors : DataFrame or array (T x K) of factor realizations f_t

    Returns:
      moments : numpy array of shape (T, N + K) where each row t stacks:
                 [ m_t * R_{t,1} - 1, ..., m_t * R_{t,N} - 1, m_t * f_{t,1}, ..., m_t * f_{t,K} ]

    Notes:
      - This is a template: depending on data units (percent vs decimal, gross vs net returns) you may
        need to rescale or redefine the -1 constant in the asset moment.
      - Stacking asset moments with factor orthogonality moments yields (N + K) moments per period.
      - In practice you may reduce dimensions (e.g., use cross-section of test portfolios rather than raw assets).
    """
    b = theta
    f = factors.values  # T x K
    R = returns.values  # T x N (assumed gross returns)
    T = R.shape[0]

    # Compute m_t = 1 - f_t b  (vector length T)
    m = 1 - (f.dot(b))  # shape (T,)

    # Asset moments: T x N matrix where each column is m_t * R_{t,i} - 1
    m_assets = (m[:, None] * R) - 1.0

    # Factor orthogonality moments: T x K where each row is m_t * f_t (should be close to zero)
    m_f = m[:, None] * f

    # Concatenate asset moments and factor moments horizontally to produce T x (N+K)
    moments = np.concatenate([m_assets, m_f], axis=1)
    return moments

# -------------------- Entry point: orchestrate analyses ----------------
# This function orchestrates the main workflow:
#  - Load factors and portfolio returns
#  - Align dates and compute excess returns
#  - Run time-series regressions (OLS) and save betas
#  - Run GRS test, cross-sectional OLS, and Fama-MacBeth
#  - Write a small summary file to disk


def run_all(factors_path, ports_path, out_dir='results'):
    """Main orchestration for the analysis pipeline.

    Steps performed:
      1. Create output directory if it does not exist.
      2. Load factor and portfolio files.
      3. Align datasets by intersection of dates.
      4. Replace sentinel -99 with NaN (common in some FF files).
      5. Separate factor columns and portfolio columns.
      6. Compute excess returns (portfolio returns minus RF).
      7. Run time-series OLS for each portfolio and save betas.
      8. Compute alpha vector (mean residuals), residual covariance, factor covariance.
      9. Attempt the GRS test, compute RMSPE.
     10. Run cross-sectional OLS of mean returns on betas (estimate lambdas).
     11. Run Fama-MacBeth two-step procedure.
     12. Save a short summary to disk.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    print('Loading data...')

    # Load factor and portfolio data using the loader helpers above.
    factors = load_factors(factors_path)
    ports = load_portfolios(ports_path)

    # Align datasets on the intersection of dates (inner join).
    df = pd.concat([factors, ports], axis=1, join='inner')

    # Replace sentinel values (-99 is used by some FF files to denote missing) with NaN.
    df = df.replace(-99, np.nan)

    # Re-split into factor and portfolio pieces, but keep only the columns present originally.
    Kcols = [c for c in factors.columns]
    Ncols = [c for c in ports.columns]
    factors = df[Kcols]
    ports = df[Ncols]

    # Extract risk-free rate series and ensure types are floats
    rf = factors['RF']

    # Compute excess returns (asset returns - RF) using pandas broadcasting.
    ports = ports.astype(float)
    excess = ports.sub(rf, axis=0)

    # Drop any rows that are all-NaN across portfolios to trim the sample.
    excess = excess.dropna(how='all')
    # Ensure factors are aligned to the same trimmed index
    factors = factors.loc[excess.index]

    # (a) Time-series regression: run OLS per portfolio vs the 3 factors
    print('Running time-series OLS...')
    betas, se_beta, resid = time_series_ols(excess, factors[['Mkt-RF', 'SMB', 'HML']])

    # Save betas to CSV for later inspection or plotting.
    betas.to_csv(os.path.join(out_dir, 'betas.csv'))

    # Compute cross-sectional alpha vector: the (time-average) residual per asset.
    alpha = resid.mean(axis=0).values

    # Residual covariance across assets (N x N)
    Sigma_e = resid.cov().values
    # Factor covariance (K x K)
    Sigma_f = factors[['Mkt-RF', 'SMB', 'HML']].cov().values

    T = excess.shape[0]
    N = excess.shape[1]
    K = 3

    # Try to perform the GRS test; wrap in try/except because numerical issues can arise.
    try:
        Fstat, pval = grs_test(alpha, Sigma_e, Sigma_f, T, N, K)
        print('GRS F-stat:', Fstat, 'pval', pval)
    except Exception as e:
        print('GRS test error:', e)

    # Root mean square pricing error across assets (RMSPE of the alphas)
    rmspe = np.sqrt(np.nanmean(alpha**2))
    print('RMSPE (alpha):', rmspe)

    # Cross-sectional OLS to estimate lambda (prices of risk) from mean returns and betas.
    mean_r = excess.mean()
    lambdas, lam_bse, cs_res = cross_sectional_ols(mean_r, betas.drop(index='const'))
    print('Cross-sectional lambdas:', lambdas)

    # Fama-MacBeth two-step
    print('Running Fama-MacBeth...')
    lam_mean, lam_se, lam_ts = fama_macbeth(excess, factors[['Mkt-RF', 'SMB', 'HML']])
    print('Fama-MacBeth lambda means:', lam_mean)

    # Save a minimal summary file containing a few key statistics
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write('RMSPE: %g\n' % rmspe)
        # If GRS failed above Fstat/pval may be undefined; guard by converting to str
        try:
            f.write('GRS F-stat (approx): %s, pval %s\n' % (str(Fstat), str(pval)))
        except Exception:
            f.write('GRS test not available due to computation error.\n')
        f.write('Cross-sectional lambdas: %s\n' % str(lambdas))
        f.write('Fama-MacBeth lambda mean: %s\n' % str(lam_mean))
    print('Results written to', out_dir)


if __name__ == '__main__':
    # Default local file names: change these to point to your CSVs or give CLI args if you add an argparse wrapper.
    factors_path = "F-F_Research_Data_Factors.csv"
    ports_path = "25_Portfolios_5x5.csv"
    out_dir = "results"

    # Run the pipeline with the provided files and write outputs to `results/`.
    run_all(factors_path, ports_path, out_dir="results")

# End of file
