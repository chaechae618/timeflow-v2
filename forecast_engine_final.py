"""
TimeFlow 예측 엔진 — 과제 완전판
모델: Naive(기준선) + ETS(Holt-Winters) + ARIMA(AutoARIMA) + STL Forecaster
지표: MAE, RMSE, SMAPE, MAPE, MASE, RSFE, TS, R²
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


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

    ljung_box_p = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        clean = values[~null_mask]
        if len(clean) >= 20:
            lb = acorr_ljungbox(clean, lags=[10], return_df=True)
            ljung_box_p = round(float(lb['lb_pvalue'].iloc[0]), 4)
    except Exception:
        pass

    clean_vals = values[~null_mask]
    skewness = round(float(stats.skew(clean_vals)), 4) if len(clean_vals) > 3 else 0
    kurtosis = round(float(stats.kurtosis(clean_vals)), 4) if len(clean_vals) > 3 else 0

    effective_null_count = original_null_count if original_null_count is not None else int(null_mask.sum())
    missing_method = "선형 보간(Linear Interpolation)" if effective_null_count > 0 else "결측치 없음"
    outlier_method = "IQR 기반 클리핑 (2.5σ)" if outlier_count > 0 else "이상치 없음"
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
        'ljung_box_p': ljung_box_p,
        'sufficient_data': n >= 30,
        'date_start': str(df[date_col].min()),
        'date_end':   str(df[date_col].max()),
        'missing_method': missing_method,
        'outlier_method': outlier_method,
        'norm_method': norm_method,
    }


class RevIN:
    def __init__(self, eps=1e-8):
        self.eps = eps
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
        v = np.clip(v, q1 - 2.5*(q3-q1), q3 + 2.5*(q3-q1))
        v_pos = v[v > 0]
        if len(v_pos) > 0 and v.max() / (v_pos.min() + 1e-10) > 100:
            self.log_transform = True
            v = np.log1p(v)
        self.mean_ = np.mean(v)
        self.std_  = np.std(v) + self.eps
        return (v - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        result = np.array(x) * self.std_ + self.mean_
        if self.log_transform:
            result = np.expm1(result)
        return result

Preprocessor = RevIN


def detect_period(values: np.ndarray, freq: str) -> int:
    defaults = {'MS': 12, 'QS': 4, 'W': 52, 'D': 7, 'H': 24}
    return defaults.get(freq, 7)


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


def compute_metrics(actual, predicted, naive_mae=None):
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
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
    if naive_mae is None:
        naive_mae = np.mean(np.abs(np.diff(a))) + 1e-10
    mase = mae / (naive_mae + 1e-10)
    rsfe = float(np.sum(res))
    mad = float(np.mean(np.abs(res))) + 1e-10
    ts = rsfe / mad
    bias_status = "편향 없음" if abs(ts) <= 4 else ("과대 예측 편향" if ts < -4 else "과소 예측 편향")
    return {
        'MAE':   round(mae, 4),
        'RMSE':  round(rmse, 4),
        'SMAPE': round(smape, 4),
        'MAPE':  round(mape, 4) if not np.isnan(mape) else 0,
        'R2':    round(r2, 4),
        'MASE':  round(mase, 4),
        'RSFE':  round(rsfe, 4),
        'TS':    round(ts, 4),
        'bias_status': bias_status,
    }


class NaiveModel:
    def __init__(self):
        self.name = 'Naive'
        self.color = '#94a3b8'
        self.train_time = 0
        self.get_metrics_cache = None

    def fit(self, values_norm, preprocessor, period=1):
        import time
        t0 = time.time()
        self.period = max(1, period)
        self.preprocessor = preprocessor
        self.values_norm = values_norm
        n = len(values_norm)
        fitted_norm = np.zeros(n)
        for i in range(n):
            fitted_norm[i] = values_norm[i - self.period] if i >= self.period else values_norm[0]
        self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
        self.train_time = round(time.time() - t0, 4)
        return self

    def predict(self, horizon):
        tail = self.values_norm[-self.period:]
        preds_norm = np.tile(tail, (horizon // self.period) + 1)[:horizon]
        return self.preprocessor.inverse_transform(preds_norm)

    def get_metrics(self, actual_orig):
        return compute_metrics(actual_orig, self.fitted_orig)


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
            m = ExponentialSmoothing(values_norm, trend='add', initialization_method='estimated')
            self.model_fit = m.fit(optimized=True)
        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.fitted_orig = preprocessor.inverse_transform(np.array(self.model_fit.fittedvalues))
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        return self.preprocessor.inverse_transform(np.array(self.model_fit.forecast(horizon)))

    def get_metrics(self, actual_orig):
        return compute_metrics(actual_orig, self.fitted_orig)


class ARIMAModel:
    def __init__(self):
        self.name = 'ARIMA'
        self.color = '#00d4ff'
        self.train_time = 0
        self.get_metrics_cache = None
        self.order = (1, 1, 1)
        self.aic = None
        self.bic = None

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
        best_aic, best_order, best_fit = np.inf, (1, d, 1), None
        for p in range(0, 3):
            for q in range(0, 2):
                try:
                    fit = ARIMA(values_norm, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order, best_fit = fit.aic, (p, d, q), fit
                except Exception:
                    continue
        self.order = best_order
        self.name = f'ARIMA{best_order}'
        self.model_fit = best_fit if best_fit else ARIMA(values_norm, order=best_order).fit()
        self.aic = round(float(self.model_fit.aic), 2)
        self.bic = round(float(self.model_fit.bic), 2)
        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.fitted_orig = preprocessor.inverse_transform(np.array(self.model_fit.fittedvalues))
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        fc = self.model_fit.forecast(steps=horizon)
        return self.preprocessor.inverse_transform(
            fc.values if hasattr(fc, 'values') else np.array(fc))

    def get_metrics(self, actual_orig):
        return compute_metrics(actual_orig, self.fitted_orig)


class STLForecastModel:
    def __init__(self):
        self.name = 'STL'
        self.color = '#a78bfa'
        self.train_time = 0
        self.get_metrics_cache = None

    def fit(self, values_norm, preprocessor, period=12):
        import time
        t0 = time.time()
        self.preprocessor = preprocessor
        self.values_norm = values_norm
        self.period = max(2, period)
        self.stl_ok = False
        try:
            from statsmodels.tsa.seasonal import STL
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            res = STL(values_norm, period=self.period, robust=True).fit()
            self.trend_ = res.trend
            self.seasonal_ = res.seasonal
            m_trend = ExponentialSmoothing(
                self.trend_, trend='add', initialization_method='estimated'
            ).fit(optimized=True)
            self.trend_model = m_trend
            self.season_pattern = self.seasonal_[-self.period:]
            fitted_norm = np.array(m_trend.fittedvalues) + self.seasonal_
            self.fitted_orig = preprocessor.inverse_transform(fitted_norm)
            self.stl_ok = True
        except Exception:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(values_norm, trend='add', initialization_method='estimated')
            fit = m.fit(optimized=True)
            self.fitted_orig = preprocessor.inverse_transform(np.array(fit.fittedvalues))
            self._fallback_fit = fit
        self.train_time = round(time.time() - t0, 2)
        return self

    def predict(self, horizon):
        if not self.stl_ok:
            return self.preprocessor.inverse_transform(
                np.array(self._fallback_fit.forecast(horizon)))
        trend_fc = np.array(self.trend_model.forecast(horizon))
        season_fc = np.tile(self.season_pattern, (horizon // self.period) + 1)[:horizon]
        return self.preprocessor.inverse_transform(trend_fc + season_fc)

    def get_metrics(self, actual_orig):
        return compute_metrics(actual_orig, self.fitted_orig)


def compute_oos_weight(model, values_orig, preprocessor, horizon_cv):
    n = len(values_orig)
    train_end = int(n * 0.8)
    if train_end + horizon_cv > n:
        return 10.0
    train = values_orig[:train_end]
    actual = values_orig[train_end:train_end + horizon_cv]
    try:
        prep_cv = RevIN()
        train_norm = prep_cv.fit_transform(train)
        period_cv = detect_period(train_norm, 'MS')
        if isinstance(model, NaiveModel):
            m_cv = NaiveModel().fit(train_norm, prep_cv, period=period_cv)
        elif isinstance(model, ETSModel):
            m_cv = ETSModel().fit(train_norm, prep_cv, period=period_cv)
        elif isinstance(model, ARIMAModel):
            m_cv = ARIMAModel().fit(train_norm, prep_cv)
        elif isinstance(model, STLForecastModel):
            m_cv = STLForecastModel().fit(train_norm, prep_cv, period=period_cv)
        else:
            return 10.0
        pred = m_cv.predict(horizon_cv)[:len(actual)]
        denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
        return float(round(np.mean(np.abs(actual - pred) / denom) * 100, 4))
    except Exception:
        return 10.0


class Ensemble:
    def __init__(self, models, oos_smapes, ci=0.90):
        self.ci = ci
        self.models = [m for m in models if not isinstance(m, NaiveModel)]
        if not self.models:
            self.models = models
        raw_weights = [1.0 / max(oos_smapes.get(m.name, 10.0), 1.0) for m in self.models]
        total = sum(raw_weights)
        self.weights = [w / total for w in raw_weights]
        self.name = f'Ensemble({len(self.models)})'

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


def compute_raw_acf(values: np.ndarray, max_lag=30) -> dict:
    n = len(values)
    centered = values - np.mean(values)
    var_v = np.var(centered) + 1e-10
    acf_vals = []
    for k in range(1, min(max_lag + 1, n)):
        c = np.mean(centered[:-k] * centered[k:]) / var_v
        acf_vals.append(round(float(c), 4))
    conf_bound = 1.96 / np.sqrt(n)
    return {'acf': acf_vals, 'conf_bound': round(conf_bound, 4)}


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
            m = ExponentialSmoothing(train, trend='add', initialization_method='estimated')
            pred = np.array(m.fit(optimized=True).forecast(horizon))[:len(actual)]
        except Exception:
            pred = np.full(len(actual), np.mean(train))
        denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
        smape = float(np.mean(np.abs(actual - pred) / denom) * 100)
        res = actual - pred
        rsfe = float(np.sum(res))
        mad = float(np.mean(np.abs(res))) + 1e-10
        ts = rsfe / mad
        results.append({
            'window': w + 1, 'train_end': train_end,
            'actual': actual, 'pred': pred,
            'smape': round(smape, 4), 'rsfe': round(rsfe, 4), 'ts': round(ts, 4),
        })
    return results


def compute_horizon_smapes(values_orig, horizons=(7, 14, 30, 60, 90)):
    n = len(values_orig)
    results = {}
    train_end = int(n * 0.75)
    for h in horizons:
        if train_end + h > n:
            break
        train = values_orig[:train_end]
        actual = values_orig[train_end:train_end + h]
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m = ExponentialSmoothing(train, trend='add', initialization_method='estimated')
            pred = np.array(m.fit(optimized=True).forecast(h))[:len(actual)]
            denom = (np.abs(actual) + np.abs(pred)) / 2 + 1e-10
            smape = float(np.mean(np.abs(actual - pred) / denom) * 100)
            results[str(h)] = round(smape, 4)
        except Exception:
            results[str(h)] = None
    return results


def generate_future_dates(last_date, freq, horizon):
    freq_map = {'H': 'h', 'D': 'D', 'W': 'W', 'MS': 'MS', 'QS': 'QS'}
    return pd.date_range(
        start=last_date, periods=horizon + 1,
        freq=freq_map.get(freq, 'D')
    )[1:]


def final_verdict(ens_metrics, acf_result, naive_smape):
    mase  = ens_metrics.get('MASE', 999)
    ts    = ens_metrics.get('TS', 999)
    smape = ens_metrics.get('SMAPE', 999)
    ljung_q = acf_result.get('ljung_box_q', 999)

    checks = {
        'mase_ok':     bool(mase < 1),
        'ts_ok':       bool(abs(ts) <= 4),
        'residual_ok': bool(ljung_q < 20),
        'smape_ok':    bool(smape < 20),
    }
    passed = sum(checks.values())
    if passed == 4:
        verdict, label = 'trust', '신뢰 가능'
    elif passed >= 2:
        verdict, label = 'conditional', '조건부 신뢰'
    else:
        verdict, label = 'review', '재검토 필요'

    messages = []
    if not checks['mase_ok']:
        messages.append(f"MASE {mase:.3f} — Naive보다 나쁨. 모델 변경 또는 시평 단축 권장")
    if not checks['ts_ok']:
        direction = "과소" if ts > 4 else "과대"
        messages.append(f"TS {ts:.2f} — {direction} 예측 편향 지속. 모델 재검토 필요")
    if not checks['residual_ok']:
        messages.append(f"Ljung-Box Q {ljung_q:.2f} — 잔차에 패턴 남아있음. 모델이 데이터를 완전히 설명하지 못함")
    if not checks['smape_ok']:
        messages.append(f"SMAPE {smape:.2f}% — 허용 오차(20%) 초과. 예측 정확도 개선 필요")
    if passed == 4:
        messages.append("모든 기준 통과. 현재 예측을 신뢰할 수 있습니다.")

    return {'verdict': verdict, 'label': label, 'passed': passed, 'total': 4,
            'checks': checks, 'messages': messages}


def run_pipeline(df, date_col, value_col,
                 horizon=12, ci=0.90, models_to_run=None,
                 original_null_count=None):

    allowed = ['naive', 'ets', 'arima', 'stl']
    if models_to_run is None:
        models_to_run = ['naive', 'ets', 'arima', 'stl']

    # 구버전 'rf' 호환
    models_to_run = ['ets' if m == 'rf' else m for m in models_to_run]
    models_to_run = [m for m in models_to_run if m in allowed]
    if not models_to_run:
        models_to_run = ['naive', 'ets', 'arima', 'stl']
    if 'naive' not in models_to_run:
        models_to_run = ['naive'] + models_to_run

    diag = diagnose(df, date_col, value_col, original_null_count=original_null_count)
    freq = diag['freq']
    n = diag['n']

    if n < 50:
        strategy_label = f'소규모 (n={n}) — 통계 모델 전용'
        models_to_run = [m for m in models_to_run if m in ['naive', 'ets', 'arima']]
    elif n < 200:
        strategy_label = f'중규모 (n={n}) — 통계+STL'
    else:
        strategy_label = f'대규모 (n={n}) — 전체 모델 풀'

    strategy = {'label': strategy_label, 'allowed_models': allowed, 'max_lags': 3}

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
    naive_mae_val = np.mean(np.abs(np.diff(values_orig))) + 1e-10

    trained_models = []

    if 'naive' in models_to_run:
        m = NaiveModel().fit(values_norm, prep, period=period)
        m.get_metrics_cache = compute_metrics(values_orig, m.fitted_orig, naive_mae=naive_mae_val)
        trained_models.append(m)

    if 'ets' in models_to_run:
        m = ETSModel().fit(values_norm, prep, period=period)
        m.get_metrics_cache = compute_metrics(values_orig, m.fitted_orig, naive_mae=naive_mae_val)
        trained_models.append(m)

    if 'arima' in models_to_run:
        m = ARIMAModel().fit(values_norm, prep)
        m.get_metrics_cache = compute_metrics(values_orig, m.fitted_orig, naive_mae=naive_mae_val)
        trained_models.append(m)

    if 'stl' in models_to_run and n >= 50:
        m = STLForecastModel().fit(values_norm, prep, period=period)
        m.get_metrics_cache = compute_metrics(values_orig, m.fitted_orig, naive_mae=naive_mae_val)
        trained_models.append(m)

    horizon_cv = min(horizon, max(3, n // 10))
    oos_smapes = {}
    for m in trained_models:
        oos_smapes[m.name] = compute_oos_weight(m, values_orig, prep, horizon_cv)

    ens = Ensemble(trained_models, oos_smapes, ci=ci)
    ens_result  = ens.predict(horizon)
    ens_fitted  = ens.get_fitted()
    ens_metrics = compute_metrics(values_orig, ens_fitted, naive_mae=naive_mae_val)

    last_date    = pd.to_datetime(df[date_col].iloc[-1])
    future_dates = generate_future_dates(last_date, freq, horizon)

    residuals  = values_orig - ens_fitted
    acf_result = compute_acf(residuals)

    backtest = rolling_backtest(values_orig, min(horizon, 12), n_windows=3)
    horizon_smapes = compute_horizon_smapes(values_orig)

    naive_pred = np.roll(values_orig, 1)
    naive_pred[0] = values_orig[0]
    denom = (np.abs(values_orig) + np.abs(naive_pred)) / 2 + 1e-10
    naive_smape = float(np.mean(np.abs(values_orig - naive_pred) / denom) * 100)

    verdict = final_verdict(ens_metrics, acf_result, naive_smape)

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
        'horizon_smapes':   horizon_smapes,
        'freq':             freq,
        'naive_smape':      round(naive_smape, 4),
        'naive_mae':        round(float(naive_mae_val), 4),
        'verdict':          verdict,
    }
