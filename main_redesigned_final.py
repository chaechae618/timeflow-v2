"""
TimeFlow — FastAPI 백엔드 (인사이트 강화 버전)
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io, sys, os, uvicorn, traceback, httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from forecast_engine_final import run_pipeline
    ENGINE_OK = True
except ImportError:
    ENGINE_OK = False

NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

app = FastAPI(title="TimeFlow API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_redesigned_final.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok", "engine": ENGINE_OK}

@app.post("/forecast")
async def forecast(
    file:     UploadFile = File(...),
    date_col: str        = Form(...),
    val_col:  str        = Form(...),
    horizon:  int        = Form(12),
    ci:       int        = Form(90),
    models:   str        = Form("naive,ets,arima,stl"),
):
    if not ENGINE_OK:
        return JSONResponse({"error": "forecast_engine_final.py를 같은 폴더에 두세요."}, status_code=500)

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        df = df.rename(columns={date_col: "ds", val_col: "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"]  = pd.to_numeric(df["y"], errors="coerce")
        df = df[["ds", "y"]].reset_index(drop=True)

        original_null_count = int(df["y"].isna().sum())
        df = df.dropna().reset_index(drop=True)

        if len(df) < 20:
            return JSONResponse({"error": "데이터가 20개 미만입니다."}, status_code=400)

        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if not model_list:
            model_list = ["naive", "ets", "arima", "stl"]

        result = run_pipeline(df, "ds", "y", horizon=horizon, ci=ci/100, models_to_run=model_list,
                              original_null_count=original_null_count)

        values_orig = df["y"].values.tolist()
        dates_orig  = [str(d)[:10] for d in df["ds"]]

        model_colors = {
            "Naive":  "#94a3b8",
            "ETS":    "#00e5a0",
            "ARIMA":  "#00d4ff",
            "STL":    "#a855f7",
        }

        model_results = []
        for m in result["models"]:
            mc = m.get_metrics_cache or {}
            model_results.append({
                "name":      m.name,
                "smape":     round(float(mc.get("SMAPE", 0)), 4),
                "mae":       round(float(mc.get("MAE",   0)), 4),
                "rmse":      round(float(mc.get("RMSE",  0)), 4),
                "r2":        round(float(mc.get("R2",    0)), 4),
                "mase":      round(float(mc.get("MASE",  0)), 4),
                "mape":      round(float(mc.get("MAPE",  0)), 4),
                "rsfe":      round(float(mc.get("RSFE",  0)), 4),
                "ts":        round(float(mc.get("TS",    0)), 4),
                "bias_status": mc.get("bias_status", "편향 없음"),
                "color":     next((v for k, v in model_colors.items() if m.name.startswith(k)), "#8a9bb8"),
                "fitted":    _safe_list(getattr(m, "fitted_orig", [])),
                "trainTime": getattr(m, "train_time", 0) or 0,
            })

        ens_metrics   = result["ensemble_metrics"]
        ens_fitted    = _safe_list(result["ensemble_fitted"])
        ens_pred      = _safe_list(result["ensemble"]["pred"])
        ens_lower     = _safe_list(result["ensemble"]["lower"])
        ens_upper     = _safe_list(result["ensemble"]["upper"])
        ens_residuals = [float(v - f) if (f is not None and v is not None) else 0.0
                         for v, f in zip(values_orig, ens_fitted)]

        # 오차 절댓값 시계열 (차트용)
        abs_errors = [abs(e) for e in ens_residuals]
        pct_errors = []
        for v, e in zip(values_orig, ens_residuals):
            if abs(v) > 1e-6:
                pct_errors.append(round(abs(e) / abs(v) * 100, 2))
            else:
                pct_errors.append(0.0)

        stl = result["stl"]
        acf_result = result.get("acf_result", {})
        raw_acf    = result.get("raw_acf", {})
        oos_smapes = {k: round(float(v), 4) for k, v in result.get("oos_smapes", {}).items()}
        diag       = result["diagnostics"]
        preprocess = result.get("preprocess_info", {})

        backtest_windows = []
        for w in result["backtest"]:
            backtest_windows.append({
                "window":     w["window"],
                "smape":      round(float(w["smape"]), 4),
                "rsfe":       round(float(w.get("rsfe", 0)), 4),
                "ts":         round(float(w.get("ts", 0)), 4),
                "cutoffDate": dates_orig[min(w["train_end"], len(dates_orig)-1)],
                "actual":     _safe_list(w["actual"]),
                "pred":       _safe_list(w["pred"]),
            })

        return {
            "ok": True,
            "dates":       dates_orig,
            "values":      values_orig,
            "futureDates": [str(d)[:10] for d in result["future_dates"]],

            "ensemble": {
                "pred":       ens_pred,
                "lower":      ens_lower,
                "upper":      ens_upper,
                "fitted":     ens_fitted,
                "residuals":  ens_residuals,
                "absErrors":  abs_errors,
                "pctErrors":  pct_errors,
                "mae":        round(float(ens_metrics.get("MAE",   0)), 4),
                "rmse":       round(float(ens_metrics.get("RMSE",  0)), 4),
                "smape":      round(float(ens_metrics.get("SMAPE", 0)), 4),
                "mape":       round(float(ens_metrics.get("MAPE",  0)), 4),
                "r2":         round(float(ens_metrics.get("R2",    0)), 4),
                "mase":       round(float(ens_metrics.get("MASE",  0)), 4),
                "rsfe":       round(float(ens_metrics.get("RSFE",  0)), 4),
                "ts":         round(float(ens_metrics.get("TS",    0)), 4),
                "bias_status": ens_metrics.get("bias_status", "편향 없음"),
            },

            "modelResults": model_results,

            "stl": {
                "trend":          _safe_list(stl["trend"]),
                "seasonal":       _safe_list(stl["seasonal"]),
                "residual":       _safe_list(stl["residual"]),
                "period":         int(stl["period"]),
                "trendStrength":  round(float(stl["trend_strength"]),  4),
                "seasonStrength": round(float(stl["season_strength"]), 4),
            },

            "backtest": backtest_windows,

            "acf": {
                "vals":         [round(float(v), 4) for v in acf_result.get("acf", [])],
                "confBound":    round(float(acf_result.get("conf_bound", 0.196)), 4),
                "ljungBoxQ":    round(float(acf_result.get("ljung_box_q", 0)), 4),
                "whiteNoise":   bool(acf_result.get("white_noise", True)),
                "nSignificant": int(acf_result.get("n_significant", 0)),
            },

            "rawAcf": {
                "vals":      [round(float(v), 4) for v in raw_acf.get("acf", [])],
                "confBound": round(float(raw_acf.get("conf_bound", 0.196)), 4),
            },

            "oosSMAPEs": oos_smapes,

            "diagnostics": {
                "n":            diag.get("n", 0),
                "nullCount":    original_null_count,
                "outlierCount": diag.get("outlier_count", 0),
                "outlierPct":   diag.get("outlier_pct", 0),
                "freq":         diag.get("freq", "unknown"),
                "mean":         diag.get("mean", 0),
                "std":          diag.get("std", 0),
                "min":          diag.get("min", 0),
                "max":          diag.get("max", 0),
                "median":       diag.get("median", 0),
                "skewness":     diag.get("skewness", 0),
                "kurtosis":     diag.get("kurtosis", 0),
                "isStationary": bool(diag.get("is_stationary")) if diag.get("is_stationary") is not None else None,
                "adfPvalue":    float(diag.get("adf_pvalue")) if diag.get("adf_pvalue") is not None else None,
                "adfStat":      float(diag.get("adf_stat")) if diag.get("adf_stat") is not None else None,
                "dateStart":    diag.get("date_start", ""),
                "dateEnd":      diag.get("date_end", ""),
                "missingMethod":  diag.get("missing_method", ""),
                "outlierMethod":  diag.get("outlier_method", ""),
                "normMethod":     diag.get("norm_method", ""),
            },

            "preprocessInfo": {
                "logApplied": preprocess.get("log_applied", False),
                "normMean":   preprocess.get("norm_mean", 0),
                "normStd":    preprocess.get("norm_std", 0),
            },

            "naiveSmape": result.get("naive_smape", 0),
            "freq":       result["freq"],
            "strategy":   result["strategy"]["label"],
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


def _safe_list(arr):
    if arr is None: return []
    try:
        out = []
        for v in arr:
            f = float(v)
            if np.isnan(f) or np.isinf(f):
                out.append(None)
            else:
                out.append(round(f, 6))
        return out
    except:
        return []


@app.get("/news")
async def get_news(keyword: str = "시계열 예측", lang: str = "ko"):
    if not NEWS_API_KEY:
        return JSONResponse({"error": "NEWS_API_KEY가 설정되지 않았습니다."}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        keyword,
                    "language": lang,
                    "sortBy":   "publishedAt",
                    "pageSize": 6,
                    "apiKey":   NEWS_API_KEY,
                }
            )
            data = resp.json()

        articles = data.get("articles", [])
        if not articles and lang == "ko":
            async with httpx.AsyncClient(timeout=10) as client:
                resp2 = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q":        keyword,
                        "language": "en",
                        "sortBy":   "publishedAt",
                        "pageSize": 6,
                        "apiKey":   NEWS_API_KEY,
                    }
                )
                data = resp2.json()
                articles = data.get("articles", [])

        result = []
        for a in articles:
            if not a.get("title") or a["title"] == "[Removed]":
                continue
            result.append({
                "title":       a.get("title", ""),
                "description": a.get("description", "") or "",
                "url":         a.get("url", ""),
                "source":      a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", "")[:10],
                "urlToImage":  a.get("urlToImage", ""),
            })

        return {"ok": True, "articles": result[:6]}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── AI 인사이트 (OpenAI API) ──────────────────────────────
@app.post("/insight")
async def get_insight(payload: dict):
    if not OPENAI_API_KEY:
        return JSONResponse({"error": "OPENAI_API_KEY가 설정되지 않았습니다."}, status_code=400)
    try:
        smape       = payload.get("smape", 0)
        mase        = payload.get("mase", 0)
        ts          = payload.get("ts", 0)
        r2          = payload.get("r2", 0)
        trend_str   = payload.get("trendStrength", 0)
        season_str  = payload.get("seasonStrength", 0)
        period      = payload.get("period", 12)
        freq        = payload.get("freq", "MS")
        horizon     = payload.get("horizon", 12)
        n           = payload.get("n", 0)
        date_start  = payload.get("dateStart", "")
        date_end    = payload.get("dateEnd", "")
        val_col     = payload.get("valCol", "값")
        naive_smape = payload.get("naiveSmape", 0)
        pred_values = payload.get("predValues", [])
        bias_status = payload.get("biasStatus", "편향 없음")
        strategy    = payload.get("strategy", "")
        data_mean   = payload.get("mean", 0)
        data_std    = payload.get("std", 0)
        data_min    = payload.get("min", 0)
        data_max    = payload.get("max", 0)
        best_model  = payload.get("bestModel", "")
        backtest_smape = payload.get("backtestSmape", 0)
        overfit_ratio  = payload.get("overfitRatio", 1.0)

        freq_label = {"MS":"월별","D":"일별","W":"주별","H":"시간별","QS":"분기별"}.get(freq, freq)

        # 종합 판정 계산
        bad_flags = sum([smape > 20, mase >= 1, abs(ts) > 4])
        if bad_flags == 0 and smape <= 10:
            overall = "신뢰 가능 (우수)"
        elif bad_flags >= 2 or smape > 20:
            overall = "재검토 필요 (미흡)"
        else:
            overall = "조건부 사용 가능 (양호)"

        pred_trend = ""
        if len(pred_values) >= 2:
            delta = pred_values[-1] - pred_values[0]
            pct   = (delta / (abs(pred_values[0]) + 1e-10)) * 100
            pred_trend = f"예측 기간 동안 {'+' if delta > 0 else ''}{pct:.1f}% {'상승' if delta > 0 else '하락'} 전망"

        overfit_note = ""
        if overfit_ratio > 3:
            overfit_note = f"⚠️ 과적합 가능성 있음 (백테스트/인샘플 SMAPE 비율: {overfit_ratio:.1f}배)"
        else:
            overfit_note = f"✅ 일반화 양호 (백테스트/인샘플 SMAPE 비율: {overfit_ratio:.1f}배)"

        prompt = f"""당신은 시계열 데이터 분석 전문가입니다.
아래 시계열 예측 결과를 분석하여 실무진이 바로 활용할 수 있는 핵심 인사이트를 한국어로 작성해주세요.
수치를 나열하는 것이 아니라, 이 데이터가 '무엇을 말하는지', '예측이 좋은지 나쁜지', '어떻게 활용해야 하는지'에 집중해주세요.

[데이터 기본 정보]
- 분석 대상: {val_col}
- 기간: {date_start} ~ {date_end} ({n}개 관측값, {freq_label} 데이터)
- 데이터 범위: {data_min:.1f} ~ {data_max:.1f} (평균 {data_mean:.1f}, 표준편차 {data_std:.1f})
- 분석 전략: {strategy}

[예측 성능 — 종합 판정: {overall}]
- SMAPE: {smape:.2f}% (나이브 기준 {naive_smape:.2f}%)
- MASE: {mase:.3f} ({'나이브보다 우수' if mase < 1 else '나이브보다 낮음'})
- R²: {r2:.3f} ({'데이터 변동의 ' + str(round(r2*100,1)) + '% 설명'})
- 편향(TS): {ts:.2f} → {bias_status}
- 최적 단일 모델: {best_model}
- 과적합 검증: {overfit_note}

[시계열 구조]
- 트렌드: {'강함' if trend_str > 0.5 else '약함'} ({trend_str:.0%})
- 계절성: {'강함' if season_str > 0.5 else '약함'} ({season_str:.0%}), 주기 {period}개({freq_label})

[예측 결과]
- 예측 기간: {horizon}{freq_label}
- {pred_trend}

다음 형식으로 작성해주세요. 각 섹션은 구체적이고 실무적으로, 2~4문장으로 작성:

## 📊 이 데이터, 한 줄로 요약하면
(데이터의 핵심 패턴을 비전문가도 이해할 수 있게 한 문장으로 + 보충 설명)

## ✅ 예측 결과, 믿어도 될까요?
(종합 판정 {overall}의 의미를 쉽게 설명. SMAPE, MASE, TS 등 핵심 지표 해석을 자연어로)

## 📈 앞으로 어떻게 될까요?
(예측 기간 동안의 방향성, 변화폭, 불확실성 수준을 실무 언어로 설명)

## ⚠️ 주의해야 할 점
(편향 방향, 계절성, 과적합 여부 등 실무에서 반드시 알아야 할 리스크)

## 💡 이렇게 활용하세요
(이 예측을 실제 업무에서 어떻게 쓸지 구체적이고 실행 가능한 제언 2~3가지)"""

        async with httpx.AsyncClient(timeout=60) as client:
            api_resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      "gpt-4o-mini",
                    "max_tokens": 1800,
                    "messages":   [
                        {"role": "system", "content": "당신은 시계열 데이터 분석 전문가입니다. 실무진이 바로 활용할 수 있는 핵심 인사이트를 한국어로 제공합니다."},
                        {"role": "user",   "content": prompt}
                    ]
                }
            )

        api_data = api_resp.json()
        print(f"[OpenAI API] status={api_resp.status_code}, response={str(api_data)[:300]}")

        # HTTP 에러 체크
        if api_resp.status_code != 200:
            err_msg = api_data.get("error", {}).get("message", f"HTTP {api_resp.status_code}")
            return JSONResponse({"error": f"OpenAI API 오류: {err_msg}"}, status_code=500)

        # content 추출
        choices = api_data.get("choices", [])
        if not choices:
            return JSONResponse({"error": "OpenAI API 응답에 choices가 없습니다."}, status_code=500)

        text = choices[0].get("message", {}).get("content", "")
        if not text:
            return JSONResponse({"error": "OpenAI API 응답 텍스트가 비어 있습니다."}, status_code=500)

        return {"ok": True, "insight": text}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    print("=" * 55)
    print("  TimeFlow API 서버 시작")
    print("  접속: http://localhost:8007")
    print("  엔진:", "✅ 준비됨" if ENGINE_OK else "❌ 없음")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8007)), reload=False)
