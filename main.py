import os
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

from sklearn.inspection import permutation_importance


app = FastAPI(
    title="CGPA Key-Factor Importance API",
    description="Remote API for computing permutation importance for the Student Academic Performance Dashboard.",
    version="1.0.0"
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "cgpa_explainer_model.joblib")
META_PATH = BASE_DIR / os.getenv("META_PATH", "cgpa_explainer_meta.json")

API_KEY = os.getenv("API_KEY", "").strip()

DEFAULT_TIME_MAP = {
    "0-1 Hour": 1,
    "1-2 Hours": 2,
    "2-3 Hours": 3,
    "More than 3 Hours": 4
}

DEFAULT_ATT_MAP = {
    "Below 40%": 1,
    "40%-59%": 2,
    "60%-79%": 3,
    "80%-100%": 4
}

DEFAULT_YESNO_MAP = {
    "No": 0,
    "Yes": 1
}

DEFAULT_FEATURE_COLS = [
    "HSC",
    "SSC",
    "Computer",
    "Preparation",
    "Gaming",
    "Attendance",
    "English",
    "Extra",
    "Job"
]


_model = None
_meta = None
_model_load_error = ""


def load_assets() -> Tuple[Any, Dict[str, Any]]:
    global _model, _meta, _model_load_error

    if _model is not None and _meta is not None:
      return _model, _meta

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")

        if not META_PATH.exists():
            raise FileNotFoundError(f"Meta file not found: {META_PATH.name}")

        _model = joblib.load(MODEL_PATH)

        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)

        _model_load_error = ""
        return _model, _meta

    except Exception as e:
        _model = None
        _meta = None
        _model_load_error = str(e)
        raise


def norm_text(value: Any) -> Any:
    if value is None:
        return np.nan

    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass

    s = str(value).strip()
    s = " ".join(s.split())
    low = s.lower()

    if low in ["0-1 hour", "0 - 1 hour", "0-1 hours", "0 - 1 hours"]:
        return "0-1 Hour"

    if low in ["1-2 hour", "1 - 2 hour", "1-2 hours", "1 - 2 hours"]:
        return "1-2 Hours"

    if low in ["2-3 hour", "2 - 3 hour", "2-3 hours", "2 - 3 hours"]:
        return "2-3 Hours"

    if low in ["more than 3 hour", "more than 3 hours", "more than 3 hrs", "more than three hours"]:
        return "More than 3 Hours"

    if low in ["yes", "y", "true", "1"]:
        return "Yes"

    if low in ["no", "n", "false", "0"]:
        return "No"

    return s


def cgpa_group(value: Any, deans_min: float, at_risk_max: float) -> Any:
    try:
        if pd.isna(value):
            return np.nan

        v = float(value)
    except Exception:
        return np.nan

    if v >= deans_min:
        return "Dean's List"

    if v < at_risk_max:
        return "At-Risk"

    return "Average"


def preprocess(
    df_raw: pd.DataFrame,
    deans_min: float,
    at_risk_max: float,
    meta: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df_raw.copy()

    feature_cols = meta.get("FEATURE_COLS", DEFAULT_FEATURE_COLS)

    if "StudentID" not in df.columns and "student_id" in df.columns:
        df = df.rename(columns={"student_id": "StudentID"})

    required_cols = list(feature_cols) + ["Overall"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError("Missing required columns for importance: " + ", ".join(missing_cols))

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    for c in ["HSC", "SSC", "Computer", "English", "Overall"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["group_label"] = df["Overall"].apply(
        lambda x: cgpa_group(x, deans_min, at_risk_max)
    )

    df = df.dropna(subset=["group_label"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows after CGPA grouping.")

    X = df[feature_cols].copy()
    y = df["group_label"].copy()

    time_map = meta.get("TIME_MAP", DEFAULT_TIME_MAP)
    att_map = meta.get("ATT_MAP", DEFAULT_ATT_MAP)
    yesno_map = meta.get("YESNO_MAP", DEFAULT_YESNO_MAP)
    impute_values = meta.get("IMPUTE_VALUES", {})

    X["Preparation"] = X["Preparation"].apply(norm_text).map(time_map)
    X["Gaming"] = X["Gaming"].apply(norm_text).map(time_map)
    X["Attendance"] = X["Attendance"].apply(norm_text).map(att_map)
    X["Extra"] = X["Extra"].apply(norm_text).map(yesno_map)
    X["Job"] = X["Job"].apply(norm_text).map(yesno_map)

    for c in ["HSC", "SSC", "Computer", "English"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    for c in feature_cols:
        if c in X.columns:
            fallback = X[c].median()

            if pd.isna(fallback):
                fallback = 0

            fill_value = impute_values.get(c, fallback)
            X[c] = X[c].fillna(fill_value)

    X = X[feature_cols].apply(pd.to_numeric, errors="coerce")

    for c in feature_cols:
        fill_value = impute_values.get(c, 0)
        X[c] = X[c].fillna(fill_value)

    return X, y


def check_api_key(request: Request) -> None:
    if API_KEY == "":
        return

    client_key = request.headers.get("X-API-Key", "").strip()

    if client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "CGPA Key-Factor Importance API",
        "endpoints": ["/health", "/compute_importance"]
    }


@app.get("/health")
def health():
    model_ready = False
    meta_ready = False
    error = ""

    try:
        load_assets()
        model_ready = _model is not None
        meta_ready = _meta is not None
    except Exception as e:
        error = str(e)

    return {
        "ok": model_ready and meta_ready,
        "model_ready": model_ready,
        "meta_ready": meta_ready,
        "model_file": MODEL_PATH.name,
        "meta_file": META_PATH.name,
        "error": error or _model_load_error
    }


@app.post("/compute_importance")
async def compute_importance(request: Request):
    try:
        check_api_key(request)

        model, meta = load_assets()

        payload = await request.json()

        deans_min = float(payload.get("deans_min", 3.67))
        at_risk_max = float(payload.get("at_risk_max", 2.00))
        records = payload.get("records", [])
        n_repeats = int(payload.get("n_repeats", 15))

        if at_risk_max >= deans_min:
            raise ValueError("Invalid thresholds: at_risk_max must be smaller than deans_min.")

        if not isinstance(records, list) or len(records) == 0:
            raise ValueError("No records provided for importance computation.")

        if n_repeats <= 0:
            n_repeats = 15

        if n_repeats > 50:
            n_repeats = 50

        df = pd.DataFrame(records)

        X, y = preprocess(
            df_raw=df,
            deans_min=deans_min,
            at_risk_max=at_risk_max,
            meta=meta
        )

        if y.nunique() < 2:
            raise ValueError(
                "Permutation importance requires at least two CGPA groups in the selected upload."
            )

        random_state = int(meta.get("random_state", 42))
        scoring = meta.get("scoring", "f1_macro")

        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1
        )

        imp = pd.DataFrame({
            "Field": X.columns,
            "Importance": result.importances_mean,
            "Std": result.importances_std
        }).sort_values("Importance", ascending=False)

        output_results = []

        for row in imp.to_dict(orient="records"):
            output_results.append({
                "Field": str(row["Field"]),
                "Importance": float(row["Importance"]),
                "Std": float(row["Std"])
            })

        return {
            "error": "",
            "results": output_results,
            "meta": {
                "rows_used": int(len(X)),
                "groups": y.value_counts().to_dict(),
                "n_repeats": int(n_repeats),
                "random_state": int(random_state),
                "scoring": str(scoring)
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "error": str(e),
                "results": []
            }
        )