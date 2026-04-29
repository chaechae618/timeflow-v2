"""
TimeFlow 예측 엔진 — 경량 배포 버전 (Render 무료 플랜 최적화)
모델: Naive + ETS + ARIMA + STL

[v6 — 안정성 강화 + 인사이트 API 연동]
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 주파수 감지
# ─────────────────────────────────────────────
def detect_frequency(date_series: pd.Series) -> str:
    if len(date_series) < 2:
        return 'unknown'
    dates = pd.to_datetime(date_series).sort_values()
    hours = dates.diff().dropna().median().total_seconds() / 3600
    if hours <= 1:     return 'H'
    elif hours <= 25:  return 'D'
    elif hours <= 170: return 'W'
    elif hours <= 800: return 'MS'
    else:              return 'QS'


# ─────────────────────────────────────────────
# 2. 데이터 진단
# ─────────────────────────────────────────────
def diagnose(df, date_col, value_col, original_null_count=None):
    values = df[value_col].values.astype(float)
    n = len(values)
    null_mask = np.isnan(values)
    q1, q3 = np.nanpercentile(values, 25), np.nanpercentile(values, 75)
    iqr = q3 - q1
    outlier_mask = (values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)
    outlier_count = int(outlier_mask.sum())
    freq = detect_frequency(df[date_col])

    is_stationary, adf_pvalue, adf_stat = None, None, None
    try:
        from statsmodels.tsa.stattools import adfuller
        clean = values[~null_mask]
        if len(clean) >= 20:
            res = adfuller(clean, autolag='AIC')
            adf_stat = round(float(res[0]), 4)
            adf_pvalue = round(float(res[1]), 4)
            is_stationary = bool(res[1] < 0.05)
    except Exception:
        pass

    clean_vals = values[~null_mask]
    skewness = round(float(stats.skew(clean_vals)), 4) if len(clean_vals) > 3 else 0
    kurtosis = round(float(stats.kurtosis(clean_vals)), 4) if len(clean_vals) > 3 else 0

    effective_null_count = original_null_count if original_null_count is not None else int(null_mask.sum())
    missing_method = "선형 보간(Linear Interpolation)" if effective_null_count > 0 else "결측치 없음"
    outlier_method = "IQR 기반 클리핑 (3.0σ)" if outlier_count > 0 else "이상치 없음"
    norm_method = "RevIN (Reversible Instance Normalization)"
    try:
        pos_vals = values[values > 0]
        if len(pos_vals) > 0 and values.max() / (pos_vals.min() + 1e-10) > 100:
            norm_method = "RevIN + 로그 변환 (스케일 100배 이상 감지)"
    except Exception:
        pass

    return {
        'n': n,
        'null_count': int(null_mask.sum()),
        'outlier_count': outlier_count,
        'outlier_pct': round(outlier_count / n * 100, 2),
        'freq': freq,
        'mean': round(float(np.nanmean(values)), 4),
        'std':  round(float(np.nanstd(values)), 4),
        'min':  round(float(np.nanmin(values)), 4),
        'max':  round(float(np.nanmax(values)), 4),
        'median': round(float(np.nanmedian(values)), 4),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'is_stationary': is_stationary,
        'adf_pvalue': adf_pvalue,
        'adf_stat': adf_stat,
        'sufficient_data': n >= 30,
        'date_start': str(df[date_col].min()),
        'date_end':   str(df[date_col].max()),
        'missing_method': missing_method,
        'outlier_method': outlier_method,
        'norm_method': norm_method,
    }


# ─────────────────────────────────────────────
# 3. RevIN 전처리
# ─────────────────────────────────────────────
class RevIN:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.fitted = False
        self.log_transform = False

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        v = values.copy().astype(float)

        nan_idx = np.where(np.isnan(v))[0]
        for i in nan_idx:
            left  = v[:i][~np.isnan(v[:i])]
            right = v[i+1:][~np.isnan(v[i+1:])]
            if len(left) and len(right):  v[i] = (left[-1] + right[0]) / 2
            elif len(left):               v[i] = left[-1]
            elif len(right):              v[i] = right[0]

        q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
        v = np.clip(v, q1 - 3.0*(q3-q1), q3 + 3.0*(q3-q1))

        v_pos = v[v > 0]
        if len(v_pos) > 0 and v.max() / (v_pos.min() + 1e-10) > 100:
            self.log_transform = True
            v = np.log1p(np.clip(v, 0, None))

        self.mean_ = np.mean(v)
        self.std_  = np.std(v) + self.eps
        self.fitted = True
        return (v - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        result = np.array(x, dtype=float) * self.std_ + self.mean_
        if self.log_transform:
            result = np.expm1(result)
        return result

Preprocessor = RevIN


# ─────────────────────────────────────────────
# 4. STL 분해
# ─────────────────────────────────────────────
def detect_period(values: np.ndarray, freq: str) -> int:
    defaults = {'MS': 12, 'QS': 4, 'W': 52, 'D': 7, 'H': 24}
    base = defaults.get(freq, 7)

    if freq in ('MS', 'QS'):
        try:
            from statsmodels.tsa.stattools import acf
            max_lag = min(base * 2, 24)
            if max_lag < base:
                return base
            acf_vals = acf(values, nlags=max_lag, fft=True)
            search_start = max(2, base // 2)
            search_end   = min(max_lag, base * 2) + 1
            region = acf_vals[search_start:search_end]
            peak_offset = int(np.argmax(region))
            detected = search_start + peak_offset
            if acf_vals[detected] < 0.10:
                return base
            return int(detected)
        except Exception:
            pass

    return base


def stl_decompose(values: np.ndarray, period: int, freq: str) -> dict:
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(values, period=max(2, period), robust=True).fit()
        trend, seasonal, residual = res.trend, res.seasonal, res.resid
    except Exception:
        n = len(values)
        w = min(15, n // 4)
        trend = np.convolve(values, np.ones(2*w+1)/(2*w+1), mode='same')
        detrended = values - trend
        seasonal = np.zeros(n)
        for p in range(period):
            idx = np.arange(p, n, period)
            seasonal[idx] = np.mean(detrended[idx])
        residual = values - trend - seasonal

    var_r = np.var(residual)
    var_d = np.var(values - trend) + 1e-10
    var_s = np.var(seasonal) + 1e-10
    return {
        'trend': trend, 'seasonal': seasonal, 'residual': residual,
        'period': period,
        'trend_strength':  max(0.0, float(1 - var_r / var_d)),
        'season_strength': max(0.0, float(1 - var_r / (var_s + var_r))),
    }


# ─────────────────────────────────────────────
# 5. 평가 지표 (rolling window TS)
# ─────────────────────────────────────────────
def compute_metrics(actual, predicted, period: int = 1):
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]

    nan_mask = np.isnan(p) | np.isinf(p)
    if nan_mask.any():
        p = p.copy()
        p[nan_mask] = a[nan_mask]

    res = a - p

    mae  = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res**2)))
    denom = (np.abs(a) + np.abs(p)) / 2 + 1e-10
    smape = float(np.mean(np.abs(res) / denom) * 100)
    nz = np.abs(a) > 1e-6
    mape = float(np.mean(np.abs(res[nz] / a[nz])) * 100) if nz.sum() > 0 else float('nan')
    ss_res = np.sum(res**2)
    ss_tot = np.sum((a - np.mean(a))**2) + 1e-10
    r2 = float(1 - ss_res / ss_tot)

    safe_period = max(1, int(period))
    if n > safe_period:
        naive_errors = np.abs(a[safe_period:] - a[:-safe_period])
        naive_mae = float(np.mean(naive_errors)) + 1e-10
    else:
        naive_mae = float(np.mean(np.abs(np.diff(a)))) + 1e-10
    mase = mae / naive_mae

    ts_window = min(n, 20)
    res_w = res[-ts_window:]
    rsfe = float(np.sum(res_w))
    mad  = float(np.mean(np.abs(res_w))) + 1e-10
    ts   = rsfe / mad

    bias_status = (
        "편향 없음" if abs(ts) <= 4
        else ("과대 예측 편향" if ts < -4 else "과소 예측 편향")
    )

    return {
        'MAE':         round(mae, 4),
        'RMSE':        round(rmse, 4),
        'SMAPE':       round(smape, 4),
        'MAPE':        round(mape, 4) if not np.isnan(mape) else 0,
        'R2':          round(r2, 4),
        'MASE':        round(mase, 4),
        'RSFE':        round(rsfe, 4),
        'TS':          round(ts, 4),
        'bias_status': bias_status,
    }


# ─────────────────────────────────────────────
# 6. ETS 모델
# ─────────────────────────────────────────────
class ETSModel:
    def __init__(self):
        self.name = 'ETS'
        self.color = '#00e5a0'
        self.train_time = 0
        self.get_metrics_cache = None

    def fit(self, values_norm, preprocessor, period=12):
        import time
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        t0 = time.time()
        n = len(values_norm)
        try:
            use_seasonal = period >= 2 and n >= period * 2
            m = ExponentialSmoothing(
                values_norm, trend='add',
                seasonal='add' if use_seasonal else None,
                seasonal_periods=period if use_seasonal else None,
                initialization_method='estimated'
            )
            self.model_fit = m.fit(optimized=True)
        except Exception:
            m = ExponentialSmoothing(
                values_norm, trend='add',
                initialization_method='estimated'
            )
            self.model_fit = m.fit(optimized=True)

        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.period = period
        self.fitted_orig = preprocessor.inverse_transform(
            np.array(self.model_fit.fittedvalues)
        )
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        return self.preprocessor.inverse_transform(
            np.array(self.model_fit.forecast(horizon))
        )

    def get_metrics(self, actual_orig, period=1):
        return compute_metrics(actual_orig, self.fitted_orig, period=period)


# ─────────────────────────────────────────────
# 7. ARIMA 모델
# ─────────────────────────────────────────────
class ARIMAModel:
    def __init__(self):
        self.name = 'ARIMA'
        self.color = '#00d4ff'
        self.train_time = 0
        self.get_metrics_cache = None
        self.order = (1, 1, 1)

    def fit(self, values_norm, preprocessor):
        import time
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        t0 = time.time()

        d = 0
        try:
            if adfuller(values_norm)[1] > 0.05:
                d = 1
        except Exception:
            d = 1

        best_aic, best_order = np.inf, (1, d, 1)
        for p in range(0, 3):
            for q in range(0, 2):
                try:
                    fit = ARIMA(values_norm, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order = fit.aic, (p, d, q)
                except Exception:
                    continue

        self.order = best_order
        self.name = f'ARIMA{best_order}'
        self.model_fit = ARIMA(values_norm, order=best_order).fit()
        self.preprocessor = preprocessor
        self.values_norm = values_norm

        fitted_norm = np.array(self.model_fit.fittedvalues)

        nan_mask = np.isnan(fitted_norm) | np.isinf(fitted_norm)
        if nan_mask.any():
            fitted_norm = fitted_norm.copy()
            fitted_norm[nan_mask] = values_norm[nan_mask]

        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        fc = self.model_fit.forecast(steps=horizon)
        fc_arr = fc.values if hasattr(fc, 'values') else np.array(fc)
        return self.preprocessor.inverse_transform(fc_arr)

    def get_metrics(self, actual_orig, period=1):
        return compute_metrics(actual_orig, self.fitted_orig, period=period)


# ─────────────────────────────────────────────
# 8. Naive 모델
# ─────────────────────────────────────────────
class NaiveModel:
    def __init__(self):
        self.name = 'Naive'
        self.color = '#94a3b8'
        self.train_time = 0
        self.get_metrics_cache = None

    def fit(self, values_norm, preprocessor, period=1):
        import time
        t0 = time.time()
        self.values_norm = values_norm
        self.preprocessor = preprocessor
        self.period = max(1, int(period))
        n = len(values_norm)
        fitted_norm = np.concatenate([
            values_norm[:self.period],
            values_norm[:-self.period]
        ])
        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        tail = self.values_norm[-self.period:]
        reps = (horizon // self.period) + 2
        repeated = np.tile(tail, reps)[:horizon]
        return self.preprocessor.inverse_transform(repeated)

    def get_metrics(self, actual_orig, period=1):
        return compute_metrics(actual_orig, self.fitted_orig, period=period)


# ─────────────────────────────────────────────
# 9. STL Forecaster
# ─────────────────────────────────────────────
class STLModel:
    def __init__(self):
        self.name = 'STL'
        self.color = '#a855f7'
        self.train_time = 0
        self.get_metrics_cache = None

    def fit(self, values_norm, preprocessor, period=7):
        import time
        t0 = time.time()
        self.values_norm = values_norm
        self.preprocessor = preprocessor
        self.period = max(2, int(period))
        n = len(values_norm)

        try:
            from statsmodels.tsa.seasonal import STL
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            stl_res = STL(values_norm, period=self.period, robust=True).fit()
            self.seasonal_ = stl_res.seasonal
            sa = values_norm - self.seasonal_
            try:
                m = ExponentialSmoothing(
                    sa, trend='add', initialization_method='estimated'
                )
                self.ets_fit = m.fit(optimized=True)
            except Exception:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                self.ets_fit = SimpleExpSmoothing(sa).fit()
            fitted_norm = np.array(self.ets_fit.fittedvalues) + self.seasonal_
        except Exception:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            try:
                m = ExponentialSmoothing(
                    values_norm, trend='add', initialization_method='estimated'
                )
                self.ets_fit = m.fit(optimized=True)
            except Exception:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                self.ets_fit = SimpleExpSmoothing(values_norm).fit()
            self.seasonal_ = np.zeros(n)
            fitted_norm = np.array(self.ets_fit.fittedvalues)

        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        pred_sa = np.array(self.ets_fit.forecast(horizon))
        n = len(self.seasonal_)
        future_seasonal = np.array([
            self.seasonal_[i % self.period] for i in range(n, n + horizon)
        ])
        pred_norm = pred_sa + future_seasonal
        return self.preprocessor.inverse_transform(pred_norm)

    def get_metrics(self, actual_orig, period=1):
        return compute_metrics(actual_orig, self.fitted_orig, period=period)


# ─────────────────────────────────────────────
# 10. OOS 가중치
# ─────────────────────────────────────────────
def compute_oos_weight(model, values_orig, preprocessor, horizon_cv, freq='MS'):
    n = len(values_orig)
    train_end = int(n * 0.8)
    if train_end + horizon_cv > n:
        return 10.0
    train = values_orig[:train_end]
    actual = values_orig[train_end:train_end + horizon_cv]
    try:
        prep_cv = RevIN()
        train_norm = prep_cv.fit_transform(train)
        period_cv = detect_period(train_norm, freq)
        if isinstance(model, NaiveModel):
            m_cv = NaiveModel().fit(train_norm, prep_cv, period=period_cv)
        elif isinstance(model, ETSModel):
            m_cv = ETSModel().fit(train_norm, prep_cv, period=period_cv)
        elif isinstance(model, ARIMAModel):
            m_cv = ARIMAModel().fit(train_norm, prep_cv)
        elif isinstance(model, STLModel):
            m_cv = STLModel().fit(train_norm, prep_cv, period=period_cv)
        else:
            return 10.0
        pred = m_cv.predict(horizon_cv)[:len(actual)]
        denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
        return float(round(np.mean(np.abs(actual - pred) / denom) * 100, 4))
    except Exception:
        return 10.0


# ─────────────────────────────────────────────
# 11. 앙상블
# ─────────────────────────────────────────────
class Ensemble:
    def __init__(self, models, oos_smapes, ci=0.90):
        self.ci = ci
        self.models = models
        raw_weights = [1.0 / max(oos_smapes.get(m.name, 10.0), 1.0) for m in models]
        total = sum(raw_weights)
        self.weights = [w / total for w in raw_weights]
        self.name = f'Ensemble({len(models)})'

    def predict(self, horizon):
        preds = np.zeros(horizon)
        for m, w in zip(self.models, self.weights):
            preds += w * m.predict(horizon)
        all_residuals = []
        for m in self.models:
            if hasattr(m, 'fitted_orig') and hasattr(m, 'values_norm'):
                orig = m.preprocessor.inverse_transform(m.values_norm)
                all_residuals.extend((np.array(m.fitted_orig) - orig).tolist())
        resid_std = np.std(all_residuals) if all_residuals else np.std(preds) * 0.1
        z = stats.norm.ppf((1 + self.ci) / 2)
        uncertainty = z * resid_std * np.sqrt(1 + np.arange(horizon) * 0.03)
        return {
            'pred':    preds,
            'lower':   preds - uncertainty,
            'upper':   preds + uncertainty,
            'weights': self.weights,
        }

    def get_fitted(self):
        n = len(self.models[0].fitted_orig)
        fitted = np.zeros(n)
        for m, w in zip(self.models, self.weights):
            fitted += w * np.array(m.fitted_orig)
        return fitted


# ─────────────────────────────────────────────
# 12. ACF 진단
# ─────────────────────────────────────────────
def compute_acf(residuals, max_lag=20):
    n = len(residuals)
    centered = residuals - np.mean(residuals)
    var_r = np.var(centered) + 1e-10
    acf_vals = [float(np.mean(centered[:-k] * centered[k:]) / var_r)
                for k in range(1, max_lag + 1)]
    conf_bound = 1.96 / np.sqrt(n)
    n_sig = sum(abs(a) > conf_bound for a in acf_vals)
    q = n * (n+2) * sum(a**2 / (n - k - 1) for k, a in enumerate(acf_vals[:10]))
    return {
        'acf': acf_vals,
        'conf_bound': conf_bound,
        'n_significant': n_sig,
        'ljung_box_q': round(q, 4),
        'white_noise': q < 20,
        'warning': f'잔차 자기상관 있음 ({n_sig}/{max_lag})' if n_sig > max_lag * 0.3 else None,
    }


# ─────────────────────────────────────────────
# 13. 원본 데이터 ACF (EDA용)
# ─────────────────────────────────────────────
def compute_raw_acf(values: np.ndarray, max_lag=30) -> dict:
    n = len(values)
    centered = values - np.mean(values)
    var_v = np.var(centered) + 1e-10
    acf_vals = []
    for k in range(1, min(max_lag + 1, n)):
        c = np.mean(centered[:-k] * centered[k:]) / var_v
        acf_vals.append(round(float(c), 4))
    conf_bound = 1.96 / np.sqrt(n)
    return {
        'acf': acf_vals,
        'conf_bound': round(conf_bound, 4),
    }


# ─────────────────────────────────────────────
# 14. 롤링 백테스트
# ─────────────────────────────────────────────
def rolling_backtest(values_orig, horizon, n_windows=3):
    n = len(values_orig)
    min_train = max(30, n // 3)
    results = []
    step = max(1, (n - horizon - min_train) // n_windows)
    for w in range(n_windows):
        train_end = min_train + w * step
        if train_end + horizon > n:
            break
        train  = values_orig[:train_end]
        actual = values_orig[train_end:train_end + horizon]
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(
                train, trend='add', initialization_method='estimated'
            )
            pred = np.array(m.fit(optimized=True).forecast(horizon))[:len(actual)]
        except Exception:
            pred = np.full(len(actual), np.mean(train))
        denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
        smape = float(np.mean(np.abs(actual - pred) / denom) * 100)
        res = actual - pred
        ts_window = min(len(res), 20)
        res_w = res[-ts_window:]
        rsfe = float(np.sum(res_w))
        mad = float(np.mean(np.abs(res_w))) + 1e-10
        ts = rsfe / mad
        results.append({
            'window': w + 1,
            'train_end': train_end,
            'actual': actual,
            'pred': pred,
            'smape': round(smape, 4),
            'rsfe': round(rsfe, 4),
            'ts': round(ts, 4),
        })
    return results


# ─────────────────────────────────────────────
# 15. 날짜 생성
# ─────────────────────────────────────────────
def generate_future_dates(last_date, freq, horizon):
    freq_map = {'H': 'h', 'D': 'D', 'W': 'W', 'MS': 'MS', 'QS': 'QS'}
    return pd.date_range(
        start=last_date,
        periods=horizon + 1,
        freq=freq_map.get(freq, 'D')
    )[1:]


# ─────────────────────────────────────────────
# 16. 메인 파이프라인
# ─────────────────────────────────────────────
def run_pipeline(df, date_col, value_col,
                 horizon=12, ci=0.90, models_to_run=None,
                 original_null_count=None):

    allowed = ['naive', 'ets', 'arima', 'stl']
    if models_to_run is None:
        models_to_run = ['naive', 'ets', 'arima', 'stl']
    models_to_run = [m for m in models_to_run if m in allowed]
    if not models_to_run:
        models_to_run = ['naive', 'ets', 'arima', 'stl']

    diag = diagnose(df, date_col, value_col, original_null_count=original_null_count)
    freq = diag['freq']
    n = diag['n']

    if n < 50:
        strategy_label = f'소규모 (n={n}) — 통계 모델 전용'
        models_to_run = [m for m in models_to_run if m in ['naive', 'ets', 'arima']]
    elif n < 200:
        strategy_label = f'중규모 (n={n}) — 통계 모델'
    else:
        strategy_label = f'대규모 (n={n}) — 전체 모델 풀'

    strategy = {
        'label': strategy_label,
        'allowed_models': allowed,
        'max_lags': 3,
    }

    prep = RevIN()
    values_orig = df[value_col].values.astype(float)
    values_norm = prep.fit_transform(values_orig)

    preprocess_info = {
        'log_applied': prep.log_transform,
        'norm_mean': round(float(prep.mean_), 4),
        'norm_std': round(float(prep.std_), 4),
        'missing_method': diag['missing_method'],
        'outlier_method': diag['outlier_method'],
        'norm_method': diag['norm_method'],
    }

    period = detect_period(values_norm, freq)
    stl_norm = stl_decompose(values_norm, period=period, freq=freq)
    stl = {
        'trend':          prep.inverse_transform(stl_norm['trend']),
        'seasonal':       stl_norm['seasonal'] * prep.std_,
        'residual':       stl_norm['residual'] * prep.std_,
        'period':         stl_norm['period'],
        'trend_strength': stl_norm['trend_strength'],
        'season_strength':stl_norm['season_strength'],
    }

    raw_acf = compute_raw_acf(values_orig)

    trained_models = []
    if 'naive' in models_to_run:
        m = NaiveModel().fit(values_norm, prep, period=period)
        m.get_metrics_cache = m.get_metrics(values_orig, period=period)
        trained_models.append(m)
    if 'ets' in models_to_run:
        m = ETSModel().fit(values_norm, prep, period=period)
        m.get_metrics_cache = m.get_metrics(values_orig, period=period)
        trained_models.append(m)
    if 'arima' in models_to_run:
        m = ARIMAModel().fit(values_norm, prep)
        m.get_metrics_cache = m.get_metrics(values_orig, period=period)
        trained_models.append(m)
    if 'stl' in models_to_run:
        m = STLModel().fit(values_norm, prep, period=period)
        m.get_metrics_cache = m.get_metrics(values_orig, period=period)
        trained_models.append(m)

    horizon_cv = min(horizon, max(3, n // 10))
    oos_smapes = {}
    for m in trained_models:
        oos_smapes[m.name] = compute_oos_weight(
            m, values_orig, prep, horizon_cv, freq=freq
        )

    ens = Ensemble(trained_models, oos_smapes, ci=ci)
    ens_result  = ens.predict(horizon)
    ens_fitted  = ens.get_fitted()
    ens_metrics = compute_metrics(values_orig, ens_fitted, period=period)

    last_date    = pd.to_datetime(df[date_col].iloc[-1])
    future_dates = generate_future_dates(last_date, freq, horizon)

    residuals  = values_orig - ens_fitted
    acf_result = compute_acf(residuals)

    backtest = rolling_backtest(values_orig, min(horizon, 12), n_windows=3)

    naive_pred = np.roll(values_orig, 1)
    naive_pred[0] = values_orig[0]
    denom = (np.abs(values_orig) + np.abs(naive_pred)) / 2 + 1e-10
    naive_smape = float(np.mean(np.abs(values_orig - naive_pred) / denom) * 100)

    return {
        'diagnostics':      diag,
        'strategy':         strategy,
        'preprocess_info':  preprocess_info,
        'preprocessor':     prep,
        'stl':              stl,
        'raw_acf':          raw_acf,
        'models':           trained_models,
        'oos_smapes':       oos_smapes,
        'ensemble':         ens_result,
        'ensemble_metrics': ens_metrics,
        'ensemble_fitted':  ens_fitted,
        'future_dates':     future_dates,
        'acf_result':       acf_result,
        'backtest':         backtest,
        'freq':             freq,
        'naive_smape':      round(naive_smape, 4),
    }
