import numpy as np
from config import ASSET_EMBEDDING_DIMS, CHALLENGES, PRICE_DATA_URL # Assume a local config.py



import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ASSETS: List[str] = [c["ticker"] for c in CHALLENGES]

# # Generate embeddings for each asset (replace with your model outputs)
# # The order must match the order in config.ASSETS
# multi_asset_embedding0 = [
#     np.random.uniform(-1, 1, size=ASSET_EMBEDDING_DIMS[asset]).tolist()
#     for asset in ASSETS
# ]
# print('multi asset embedding0----------', multi_asset_embedding0)



# =========================
# Configuration constants
# =========================

# 1 hour ahead with 3-minute sampling = 20 steps
HORIZON_STEPS = 20
RESAMPLE_RULE = "3min"

# Minimum number of supervised samples required to train
MIN_TRAIN_SAMPLES = 5

# =========================
# Database utilities
# =========================

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ts TEXT NOT NULL,
            asset TEXT NOT NULL,
            price REAL NOT NULL,
            PRIMARY KEY (ts, asset)
        )
    """)
    conn.commit()
    return conn

def insert_prices(conn: sqlite3.Connection, ts_iso: str, prices: Dict[str, float]) -> None:
    rows = [(ts_iso, asset, float(price)) for asset, price in prices.items() if asset in ASSETS]
    if not rows:
        return
    with conn:
        conn.executemany("INSERT OR IGNORE INTO prices (ts, asset, price) VALUES (?, ?, ?)", rows)

def load_asset_series(conn: sqlite3.Connection, asset: str) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT ts, price FROM prices WHERE asset = ? ORDER BY ts ASC",
        conn,
        params=[asset],
    )
    if df.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df["ts"], utc=True)
    s = pd.Series(df["price"].to_numpy(dtype=float), index=idx).sort_index()
    # Resample to 3-minute grid for consistent modeling; forward-fill/interpolate mild gaps
    s = s.resample(RESAMPLE_RULE).last()
    s = s.interpolate(method="time").ffill().bfill()
    return s

# =========================
# Fetch utilities
# =========================

def fetch_latest(url: str) -> Tuple[str, Dict[str, float]]:
    """Return (timestamp_iso_utc, prices_dict)."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Input example:
    # {"timestamp": "2025-09-23T18:34:18.450653", "prices": {"BTC": 111961.25, ...}}
    ts_raw = data.get("timestamp")
    prices = data.get("prices", {})
    if not ts_raw or not isinstance(prices, dict):
        raise ValueError("Malformed response: missing timestamp/prices")
    # Normalize timestamp to ISO UTC
    ts = pd.to_datetime(ts_raw, utc=True)
    ts_iso = ts.isoformat()
    return ts_iso, {k: float(v) for k, v in prices.items()}

# =========================
# Modeling utilities
# =========================

def _build_supervised_from_series(s: pd.Series, horizon_steps: int) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Create a supervised learning frame:
      - Features: lagged returns and simple moving stats.
      - Target: future log return over 'horizon_steps' ahead.
    """
    if s.size < 50:
        return pd.DataFrame(), pd.Series(dtype=float), s.index

    logp = np.log(s)
    ret = logp.diff()

    feats = pd.DataFrame(index=s.index)
    # lagged returns
    lags = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40]
    for L in lags:
        feats[f"ret_lag_{L}"] = ret.shift(L)

    # rolling statistics of returns
    feats["ret_ma_5"] = ret.rolling(5).mean()
    feats["ret_ma_10"] = ret.rolling(10).mean()
    feats["ret_std_20"] = ret.rolling(20).std()
    feats["ret_std_40"] = ret.rolling(40).std()

    # momentum-ish feature: log price slope over short window
    feats["logp_slope_10"] = logp.diff().rolling(10).mean()

    # Target: future cumulative return over horizon
    fut_logp = logp.shift(-horizon_steps)
    y = fut_logp - logp  # log return between t and t+h

    # Drop rows with NaNs
    data = pd.concat([feats, y.rename("y")], axis=1).dropna()
    if data.empty:
        return pd.DataFrame(), pd.Series(dtype=float), s.index

    X = data.drop(columns=["y"])
    y = data["y"]
    return X, y, data.index

def train_models_for_asset(s: pd.Series, horizon_steps: int) -> Tuple[Pipeline, Pipeline, pd.DataFrame, pd.Series, float]:
    """
    Train regression (future log return) and classification (direction) models.
    Returns:
      reg_model, clf_model, X, y, last_price
    """
    X, y, idx = _build_supervised_from_series(s, horizon_steps)
    if X.shape[0] < MIN_TRAIN_SAMPLES:
        return None, None, X, y, float(s.iloc[-1])

    # Regression: predict y (future log return)
    reg_model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0, random_state=42)),
    ])
    reg_model.fit(X, y)

    # Classification: predict sign (up vs down)
    y_dir = (y > 0).astype(int)
    clf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    # If all one class (rare early), skip training
    if y_dir.nunique() < 2:
        clf_model = None
    else:
        clf_model.fit(X, y_dir)

    return reg_model, clf_model, X, y, float(s.iloc[-1])

def predict_for_asset(asset: str, s: pd.Series, reg_model: Pipeline, clf_model: Pipeline, horizon_steps: int) -> Dict[str, float]:
    """
    Generate predictions for an asset given its series and trained models.
    Fallbacks if models are None or not enough data.
    """
    last_price = float(s.iloc[-1])
    X_live, _, _ = _build_supervised_from_series(s, horizon_steps)
    if X_live.empty:
        return {
            "pred_price": last_price,
            "pred_return": 0.0,
            "p_up": 0.5,
            "p_down": 0.5,
            "last_price": last_price,
        }

    x_latest = X_live.iloc[[-1]]  # keep as DataFrame
    if reg_model is None:
        pred_ret = 0.0
    else:
        pred_ret = float(reg_model.predict(x_latest)[0])

    pred_price = float(last_price * math.exp(pred_ret))

    if clf_model is None:
        p_up = 0.5
    else:
        proba = clf_model.predict_proba(x_latest)[0]
        # proba = [p_down, p_up] in scikit by default for binary with classes [0,1]
        p_up = float(proba[1])
    p_down = float(1.0 - p_up)

    return {
        "pred_price": pred_price,
        "pred_return": pred_ret,
        "p_up": p_up,
        "p_down": p_down,
        "last_price": last_price,
    }

# =========================
# Embedding utilities
# =========================

def _tanh_clip(x: np.ndarray, scale: float = 10.0) -> np.ndarray:
    return np.tanh(np.asarray(x, dtype=float) * scale)

def build_embeddings(
    assets: List[str],
    dims: Dict[str, int],
    predictions: Dict[str, Dict[str, float]],
    histories: Dict[str, pd.Series],
) -> List[List[float]]:
    """
    Build multi-asset embeddings per CHALLENGES dims.

    For dim == 2: [p_down, p_up]
    For dim > 2 (e.g., BTC=100):
      Start with [tanh(pred_return*10), 2*p_up - 1] in [-1,1],
      then fill with tanh-scaled recent returns up to needed length,
      then pad with zeros if short.
    """
    out: List[List[float]] = []
    for asset in assets:
        dim = int(dims[asset])
        pred = predictions.get(asset, {})

        last_price = float(pred.get("last_price", 0))
        pred_price = float(pred.get("pred_price", 0))

        vec = [float(np.sign(pred_price - last_price))] * dim

        out.append(vec)
    return out

# =========================
# Main loop
# =========================

def run_once(conn: sqlite3.Connection, url: str, output_dir: str) -> None:
    ts_iso, prices = fetch_latest(url)
    insert_prices(conn, ts_iso, prices)

    # Load histories
    histories: Dict[str, pd.Series] = {}
    for asset in ASSETS:
        histories[asset] = load_asset_series(conn, asset)

    # Train and predict per asset
    predictions: Dict[str, Dict[str, float]] = {}
    for asset in ASSETS:
        s = histories[asset]
        reg_model, clf_model, _, _, _ = train_models_for_asset(s, HORIZON_STEPS)
        pred = predict_for_asset(asset, s, reg_model, clf_model, HORIZON_STEPS)
        predictions[asset] = pred

    # Build embeddings
    multi_asset_embedding = build_embeddings(ASSETS, ASSET_EMBEDDING_DIMS, predictions, histories)

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    # predictions.json
    out_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_minutes": 60,
        "sampling_minutes": 3,
        "assets": ASSETS,
        "predictions": predictions,
        "ASSET_EMBEDDING_DIMS": ASSET_EMBEDDING_DIMS,
        "CHALLENGES": CHALLENGES,
        "multi_asset_embedding": multi_asset_embedding,
    }
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # embeddings.py for easy Python import of exactly the format you showed
    emb_py = f"""# Auto-generated. Do not edit by hand.
import numpy as np

ASSETS = {repr(ASSETS)}
ASSET_EMBEDDING_DIMS = {repr(ASSET_EMBEDDING_DIMS)}
CHALLENGES = {repr(CHALLENGES)}

multi_asset_embedding = {repr(multi_asset_embedding)}
"""
    with open(os.path.join(output_dir, "embeddings.py"), "w", encoding="utf-8") as f:
        f.write(emb_py)

    # Console summary
    lines = []
    for a in ASSETS:
        p = predictions[a]
        lines.append(
            f"{a:7s} last={p['last_price']:.6f}  pred_1h={p['pred_price']:.6f}  "
            f"ret={p['pred_return']:.6f}  p_up={p['p_up']:.3f}"
        )
    print("[OK] Fetched, stored, trained, predicted.")
    print("     " + "\n     ".join(lines))
    print(f"     Wrote: {os.path.join(output_dir, 'predictions.json')}")
    print(f"     Wrote: {os.path.join(output_dir, 'embeddings.py')}")

    return multi_asset_embedding


### Step 2
import json
import time
import secrets
import requests
from timelock import Timelock

# Your Bittensor hotkey
my_hotkey = "5Hme13v5gL5CYA5fZuYTdwYpTVwdPenVZ3sDoQ8gX2qoWrmx" # <-- REPLACE WITH YOUR HOTKEY

# Drand beacon configuration (do not change)
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

# Fetch beacon info to calculate a future round
info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
print('info---------', info)


### Step 3
# The filename can be anything, but using the hotkey is good practice.

try:
    filename = f"../public_server/{my_hotkey}" 

    conn = init_db('prices.db')

    while True:
        last_time = time.time()
        try:
            future_time = last_time + 30  # Target a round ~30 seconds in the future
            target_round = int((future_time - info["genesis_time"]) // info["period"])

            multi_asset_embedding = run_once(conn, PRICE_DATA_URL, 'out')
            print('multi asset embedding----------', multi_asset_embedding)

            # Create the plaintext by joining embeddings and the hotkey
            plaintext = f"{str(multi_asset_embedding)}:::{my_hotkey}"

            # Encrypt the plaintext for the target round
            tlock = Timelock(DRAND_PUBLIC_KEY)
            salt = secrets.token_bytes(32)
            ciphertext_hex = tlock.tle(target_round, plaintext, salt).hex()

            payload = {
                "round": target_round,
                "ciphertext": ciphertext_hex,
            }

            with open(filename, "w") as f:
                json.dump(payload, f)

        except Exception as e:
            sys.stderr.write(f"[ERRO] {e}\n")        

        now = time.time()
        if (last_time + 180 > now):
            time.sleep(last_time + 180 - now)
        else:
            time.sleep(1)
    
except Exception as e:
    sys.stderr.write(f"[WARN] {e}\n")



