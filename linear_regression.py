# linear_regression.py
# Simple contextual linear-regression baseline with train/val/test split.
# - Collects on TRAIN with epsilon-greedy (random when no model yet)
# - Fits ridge regression to predict immediate reward
# - Evaluates greedily on TRAIN/VAL/TEST each epoch
#
# Outputs:
#   - linreg_epoch_log.csv  (epoch-level metrics for train/val/test)
#   - linreg_theta.csv      (final learned coefficients)

import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ====================== CONFIG ======================
# If you have split CSVs from build_sessions_split.py they'll be used.
SESSIONS_TRAIN = "sessions_train.csv"
SESSIONS_VAL   = "sessions_validation.csv"
SESSIONS_TEST  = "sessions_test.csv"

# Fallback (if the split files don't exist, this single file will be used for all sets)
SESSIONS_CSV_FALLBACK = "sessions_2024.csv"

PRICES_CSV = "dayahead_nl_2024_filled.csv"
TZ_LOCAL   = "Europe/Amsterdam"

# Date window used to build a complete price index (keeps env stable)
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

DT_MIN = 15
DT_H   = DT_MIN / 60.0
SITE_PMAX_KW = 11.0 * 1

# Objective / demand
ALPHA = 2.0
RETAIL_ACTIONS = [round(x, 2) for x in np.arange(0.10, 1.00 + 0.001, 0.05)]  # €0.10..€1.00
REF_PRICE   = 0.35
ELASTICITY  = 1.5
URG_ETA     = 0.25

# Training schedule
EPOCHS = 30
EPSILON_EXPLORE = 0.30
EPSILON_DECAY   = 0.98
RNG_SEED = 42

# Regression
RIDGE_L2 = 1e-4
MAX_SAMPLES = 500_000
# ====================================================

rng = np.random.default_rng(RNG_SEED)
PRICE_QS = None  # used in env._state()

# ---------- Common IO ----------
def load_sessions(path: str) -> pd.DataFrame:
    s = pd.read_csv(path)
    for c in ("arrival", "departure", "energy_kWh"):
        if c not in s.columns:
            raise ValueError(f"{path} mist kolom '{c}'")

    arr_utc = pd.to_datetime(s["arrival"].astype(str).str.strip(), errors="coerce", utc=True)
    dep_utc = pd.to_datetime(s["departure"].astype(str).str.strip(), errors="coerce", utc=True)

    # If parsing failed a lot, try as local then convert to UTC
    if arr_utc.isna().mean() > 0.5 or dep_utc.isna().mean() > 0.5:
        arr_local = pd.to_datetime(s["arrival"].astype(str).str.strip(), errors="coerce")
        dep_local = pd.to_datetime(s["departure"].astype(str).str.strip(), errors="coerce")
        if getattr(arr_local.dt, "tz", None) is None:
            arr_local = arr_local.dt.tz_localize(TZ_LOCAL)
        else:
            arr_local = arr_local.dt.tz_convert(TZ_LOCAL)
        if getattr(dep_local.dt, "tz", None) is None:
            dep_local = dep_local.dt.tz_localize(TZ_LOCAL)
        else:
            dep_local = dep_local.dt.tz_convert(TZ_LOCAL)
        arr_utc = arr_local.dt.tz_convert("UTC")
        dep_utc = dep_local.dt.tz_convert("UTC")

    s["arrival_utc"]   = arr_utc
    s["departure_utc"] = dep_utc

    if "peak_kW" in s.columns:
        s["pmax_kW"] = pd.to_numeric(s["peak_kW"], errors="coerce").fillna(11.0).clip(lower=0.0)
    elif "pmax_kW" in s.columns:
        s["pmax_kW"] = pd.to_numeric(s["pmax_kW"], errors="coerce").fillna(11.0).clip(lower=0.0)
    else:
        s["pmax_kW"] = 11.0

    s["energy_kWh"] = pd.to_numeric(s["energy_kWh"], errors="coerce")
    s = s.dropna(subset=["arrival_utc", "departure_utc", "energy_kWh"])
    s = s[s["energy_kWh"] > 1e-6].copy()
    if "cp_id" not in s.columns:
        s["cp_id"] = "cp0"
    return s[["cp_id", "arrival_utc", "departure_utc", "energy_kWh", "pmax_kW"]].reset_index(drop=True)

def load_prices(path=PRICES_CSV) -> pd.Series:
    df = pd.read_csv(path)

    # --- timestamps → UTC index ---
    if "timestamp_europe_amsterdam" in df.columns:
        s = df["timestamp_europe_amsterdam"].astype(str).str.strip()
        ts_utc = pd.to_datetime(s, errors="coerce", utc=True)
    elif "timestamp_local" in df.columns:
        s = df["timestamp_local"].astype(str).str.strip()
        ts_utc = pd.to_datetime(s, errors="coerce", utc=True)
        if ts_utc.isna().mean() > 0.5:
            ts_local = pd.to_datetime(s, errors="coerce")
            if getattr(ts_local.dt, "tz", None) is None:
                ts_local = ts_local.dt.tz_localize(TZ_LOCAL)
            else:
                ts_local = ts_local.dt.tz_convert(TZ_LOCAL)
            ts_utc = ts_local.dt.tz_convert("UTC")
    elif "timestamp_utc" in df.columns:
        ts_utc = pd.to_datetime(df["timestamp_utc"].astype(str).str.strip(), errors="coerce", utc=True)
    else:
        raise ValueError("Kon geen tijdkolom in prijzen CSV vinden.")

    # --- price to EUR/kWh ---
    def to_num(x):
        return pd.to_numeric(x.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")

    cand = {}
    if "price_eur_per_kwh" in df.columns:
        cand["kwh"] = to_num(df["price_eur_per_kwh"])
    if "price_eur_per_mwh" in df.columns:
        cand["mwh"] = to_num(df["price_eur_per_mwh"]) / 1000.0
    if "price" in df.columns:
        raw = to_num(df["price"])
        factor = 1.0
        if "unit" in df.columns:
            unit = df["unit"].astype(str).str.upper()
            factor = np.where(unit.str.contains("MWH"), 1/1000.0, 1.0)
        cand["price+unit"] = raw * factor

    best, nn = None, -1
    for ser in cand.values():
        k = int(ser.notna().sum())
        if k > nn:
            best, nn = ser, k
    if best is None or nn == 0:
        raise ValueError("Geen bruikbare prijzen gevonden in prijzen CSV.")

    prices = (pd.DataFrame({"eur_per_kwh": best.values}, index=ts_utc)
              .dropna(subset=["eur_per_kwh"])
              .sort_index())
    prices = prices.groupby(level=0)["eur_per_kwh"].mean().to_frame()

    pr15 = prices["eur_per_kwh"].resample("15min").ffill()

    start_utc = pd.Timestamp(START_DATE, tz=TZ_LOCAL).tz_convert("UTC") - pd.Timedelta(hours=1)
    end_utc   = pd.Timestamp(END_DATE,   tz=TZ_LOCAL).tz_convert("UTC")
    full_idx  = pd.date_range(start_utc, end_utc, freq="15min", tz="UTC")
    pr15      = pr15.reindex(full_idx).ffill().bfill()

    if pr15.dropna().empty:
        raise ValueError("Prijsreeks blijft leeg na vullen.")
    return pr15

def daterange_local_days(start_date, end_date):
    start = pd.Timestamp(start_date, tz=TZ_LOCAL)
    end   = pd.Timestamp(end_date, tz=TZ_LOCAL)
    cur = start
    while cur < end:
        yield cur
        cur += pd.Timedelta(days=1)

# ---------- Demand model ----------
def willingness(price):
    x = (REF_PRICE - price) * ELASTICITY
    return 1.0 / (1.0 + np.exp(-x))

def desired_power_kW(price, pmax, hours_left, energy_left_kWh):
    w = willingness(price)
    base = w * pmax
    req = 0.0
    if hours_left > 1e-6:
        req = min(pmax, energy_left_kWh / hours_left / DT_H * DT_H)
    urg_floor = URG_ETA * pmax
    want = max(base, req, urg_floor if energy_left_kWh > 0 else 0.0)
    return float(min(pmax, want))

# ---------- Environment ----------
class PricingEnv:
    def __init__(self, sessions_df, price_series_utc, site_pmax=SITE_PMAX_KW):
        self.sessions = sessions_df.copy()
        self.pr = price_series_utc.copy()  # UTC indexed Series
        self.site_pmax = site_pmax

    def _price_at(self, t_utc):
        idx = self.pr.index
        pos = idx.searchsorted(t_utc)
        if pos < len(idx) and idx[pos] == t_utc:
            return float(self.pr.iloc[pos])
        if pos == 0:
            return float(self.pr.iloc[0])
        return float(self.pr.iloc[pos - 1])

    def reset(self, day_local):
        self.t_local = day_local if getattr(day_local, "tz", None) is not None else pd.Timestamp(day_local, tz=TZ_LOCAL)
        self.t_end_local = self.t_local + pd.Timedelta(days=1)
        self.t_utc  = self.t_local.tz_convert("UTC")
        self.t_end_utc = self.t_end_local.tz_convert("UTC")

        s = self.sessions[
            (self.sessions["arrival_utc"] < self.t_end_utc) &
            (self.sessions["departure_utc"] > self.t_utc)
        ].copy()

        total_duration = (s["departure_utc"] - s["arrival_utc"]).dt.total_seconds().clip(lower=1.0)
        a = s["arrival_utc"].clip(lower=self.t_utc, upper=self.t_end_utc)
        d = s["departure_utc"].clip(lower=self.t_utc, upper=self.t_end_utc)
        frac = (d - a).dt.total_seconds().values / total_duration.values
        s["energy_left_kWh"] = s["energy_kWh"].values * np.clip(frac, 0, 1)

        self.jobs = s.reset_index(drop=True)
        self.done = False
        self.cum_revenue = 0.0
        self.cum_peak_kWh = 0.0
        return self._state()

    def step(self, retail_price):
        if self.done:
            return self._state(), 0.0, True, {}

        mask = (self.jobs["arrival_utc"] <= self.t_utc) & (self.jobs["departure_utc"] > self.t_utc) & (self.jobs["energy_left_kWh"] > 1e-9)
        powers = np.zeros(len(self.jobs), dtype=float)

        wholesale = self._price_at(self.t_utc)

        for idx in np.where(mask)[0]:
            row = self.jobs.loc[idx]
            hours_left = max((row["departure_utc"] - self.t_utc).total_seconds()/3600.0, DT_H)
            powers[idx] = desired_power_kW(retail_price, row["pmax_kW"], hours_left, row["energy_left_kWh"])

        total_power = powers.sum()
        if total_power > self.site_pmax and total_power > 0:
            powers *= self.site_pmax / total_power
            total_power = self.site_pmax

        e_delivered = powers * DT_H
        self.jobs.loc[:, "energy_left_kWh"] = np.maximum(self.jobs["energy_left_kWh"] - e_delivered, 0.0)

        margin = max(retail_price - wholesale, -1.0)
        revenue = float(e_delivered.sum() * margin)
        local_hour = self.t_utc.tz_convert(TZ_LOCAL).hour
        peak_kWh = float(total_power * DT_H) if 16 <= local_hour < 20 else 0.0
        reward = revenue - ALPHA * peak_kWh

        self.t_utc += pd.Timedelta(minutes=DT_MIN)
        if self.t_utc >= self.t_end_utc:
            self.done = True

        self.cum_revenue += revenue
        self.cum_peak_kWh += peak_kWh
        return self._state(), reward, self.done, {"revenue": revenue, "peak_kWh": peak_kWh}

    def _state(self):
        # slot in local quarter-hour steps
        tl = self.t_utc.tz_convert(TZ_LOCAL)
        slot = int(((tl - tl.normalize()).total_seconds() // (DT_MIN * 60)) % int(24 * 60 / DT_MIN))

        mask = (self.jobs["arrival_utc"] <= self.t_utc) & (self.jobs["departure_utc"] > self.t_utc) & (self.jobs["energy_left_kWh"] > 1e-9)
        n_act = int(mask.sum()); n_act_bin = min(n_act, 3)

        w = self._price_at(self.t_utc)
        global PRICE_QS
        if PRICE_QS is None:
            qs = np.quantile(self.pr.dropna().values, [0.25, 0.5, 0.75])
            PRICE_QS = tuple(qs)
        wbin = 0 + (w > PRICE_QS[0]) + (w > PRICE_QS[1]) + (w > PRICE_QS[2])

        urg_bin = 0
        if n_act > 0:
            hleft = (self.jobs.loc[mask, "departure_utc"] - self.t_utc).dt.total_seconds()/3600.0
            eleft = self.jobs.loc[mask, "energy_left_kWh"].values
            ratio = float(np.mean(eleft / np.maximum(1.0, hleft)))
            urg_bin = 0 if ratio < 2 else 1 if ratio < 5 else 2 if ratio < 8 else 3

        return (slot, n_act_bin, wbin, urg_bin)

# ---------- Features ----------
FEATURE_NAMES = [
    "bias",
    "price", "price2",
    "wholesale",
    "peak16_20",
    "n_act_bin", "urg_bin",
    "sin_t", "cos_t",
    "price*peak", "price*wholesale"
]

def build_features(env: PricingEnv, state, price: float) -> np.ndarray:
    slot, n_act_bin, wbin, urg_bin = state
    tloc = env.t_utc.tz_convert(TZ_LOCAL)
    hour = tloc.hour + tloc.minute/60.0
    peak = 1.0 if 16 <= tloc.hour < 20 else 0.0
    wholesale = env._price_at(env.t_utc)

    x = np.array([
        1.0,
        price, price * price,
        wholesale,
        peak,
        float(n_act_bin), float(urg_bin),
        math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
        price * peak, price * wholesale
    ], dtype=float)
    return x

# ---------- Linear Reward Model ----------
class LinearRewardModel:
    def __init__(self, n_features, l2=RIDGE_L2):
        self.theta = np.zeros(n_features, dtype=float)
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray):
        XtX = X.T @ X
        n = XtX.shape[0]
        XtX = XtX + self.l2 * np.eye(n)
        Xty = X.T @ y
        self.theta = np.linalg.solve(XtX, Xty)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.theta

# ---------- Dataset collection / evaluation ----------
def collect_dataset(env: PricingEnv, epsilon: float, buffer_X: np.ndarray, buffer_y: np.ndarray, model: LinearRewardModel | None):
    """
    Walk all days; collect (X,y) on TRAIN env.
    If model is None -> random actions.
    Else -> epsilon-greedy on predicted immediate reward.
    """
    steps = 0
    n_feat = len(FEATURE_NAMES)
    if buffer_X.size == 0:
        buffer_X = np.zeros((0, n_feat), dtype=float)

    for day_local in daterange_local_days(START_DATE, END_DATE):
        s = env.reset(day_local)
        done = False
        while not done:
            if (model is None) or (rng.random() < epsilon):
                price = float(rng.choice(RETAIL_ACTIONS))
            else:
                best_p, best_pred = None, -1e18
                for p in RETAIL_ACTIONS:
                    x_tmp = build_features(env, s, p).reshape(1, -1)
                    pred = float(model.predict(x_tmp)[0])
                    if pred > best_pred:
                        best_pred, best_p = pred, p
                price = best_p

            x = build_features(env, s, price)
            s2, reward, done, _info = env.step(price)

            buffer_X = np.vstack([buffer_X, x])
            buffer_y = np.concatenate([buffer_y, np.array([reward], dtype=float)])
            steps += 1

            # cap
            if buffer_X.shape[0] > MAX_SAMPLES:
                keep = MAX_SAMPLES // 2
                buffer_X = buffer_X[-keep:, :]
                buffer_y = buffer_y[-keep:]

            s = s2

    return buffer_X, buffer_y, steps

def eval_env(env: PricingEnv, model: LinearRewardModel):
    sum_loss = sum_rev = sum_peak = 0.0
    n_days = 0
    for day_local in daterange_local_days(START_DATE, END_DATE):
        s = env.reset(day_local)
        done = False
        day_rev = 0.0
        day_peak = 0.0
        while not done:
            best_p, best_pred = None, -1e18
            for p in RETAIL_ACTIONS:
                x = build_features(env, s, p).reshape(1, -1)
                pred = float(model.predict(x)[0])
                if pred > best_pred:
                    best_pred, best_p = pred, p
            s2, reward, done, info = env.step(best_p)
            s = s2
            day_rev  += info.get("revenue", 0.0)
            day_peak += info.get("peak_kWh", 0.0)
        day_loss = -day_rev + ALPHA * day_peak
        sum_loss += day_loss
        sum_rev  += day_rev
        sum_peak += day_peak
        n_days   += 1

    if n_days == 0:
        return float("nan"), float("nan"), float("nan")
    return sum_loss / n_days, sum_rev / n_days, sum_peak / n_days

# ---------- Training ----------
def train():
    # Load prices once
    prices = load_prices(PRICES_CSV)

    # Load session splits (or fallback)
    have_split = Path(SESSIONS_TRAIN).exists() and Path(SESSIONS_VAL).exists() and Path(SESSIONS_TEST).exists()
    if have_split:
        print("✅ Splits gevonden: sessions_train.csv, sessions_validation.csv, sessions_test.csv")
        s_train = load_sessions(SESSIONS_TRAIN)
        s_val   = load_sessions(SESSIONS_VAL)
        s_test  = load_sessions(SESSIONS_TEST)
    else:
        print("⚠️  Splits niet gevonden. Gebruik fallback:", SESSIONS_CSV_FALLBACK, "voor train/val/test allemaal.")
        s_all = load_sessions(SESSIONS_CSV_FALLBACK)
        s_train = s_all.copy()
        s_val   = s_all.copy()
        s_test  = s_all.copy()

    # Build environments
    env_train = PricingEnv(s_train, prices, site_pmax=SITE_PMAX_KW)
    env_val   = PricingEnv(s_val,   prices, site_pmax=SITE_PMAX_KW)
    env_test  = PricingEnv(s_test,  prices, site_pmax=SITE_PMAX_KW)

    n_feat = len(FEATURE_NAMES)
    model = LinearRewardModel(n_features=n_feat, l2=RIDGE_L2)

    # Replay buffer
    X_buf = np.empty((0, n_feat), dtype=float)
    y_buf = np.empty((0,), dtype=float)

    # For epsilon-greedy collection
    epsilon = EPSILON_EXPLORE
    current_model: LinearRewardModel | None = None

    rows = []  # epoch log

    for epoch in range(1, EPOCHS + 1):
        # --- Collect on TRAIN (epsilon-greedy; random while model is None) ---
        X_buf, y_buf, steps = collect_dataset(env_train, epsilon, X_buf, y_buf, model=current_model)

        # --- Fit on TRAIN buffer ---
        model.fit(X_buf, y_buf)

        # --- Eval (greedy) ---
        tr_loss, tr_rev, tr_peak = eval_env(env_train, model)
        va_loss, va_rev, va_peak = eval_env(env_val,   model)
        te_loss, te_rev, te_peak = eval_env(env_test,  model)

        rows.append({
            "epoch": epoch,
            "avg_loss_train": tr_loss, "avg_revenue_train": tr_rev, "avg_peak_kWh_train": tr_peak,
            "avg_loss_val":   va_loss, "avg_revenue_val":   va_rev, "avg_peak_kWh_val":   va_peak,
            "avg_loss_test":  te_loss, "avg_revenue_test":  te_rev, "avg_peak_kWh_test":  te_peak,
            "buffer_steps":   int(len(y_buf)),
            "epsilon":        float(epsilon),
        })

        print(f"== EPOCH {epoch}/{EPOCHS} == "
              f"train: loss={tr_loss:.4f} rev={tr_rev:.2f} peak={tr_peak:.2f} | "
              f"val: loss={va_loss:.4f} rev={va_rev:.2f} peak={va_peak:.2f} | "
              f"test: loss={te_loss:.4f} rev={te_rev:.2f} peak={te_peak:.2f} | "
              f"data={len(y_buf)} steps eps={epsilon:.3f}")

        # decay epsilon and update current_model
        epsilon = max(0.05, epsilon * EPSILON_DECAY)
        current_model = model

    # Save logs
    df = pd.DataFrame(rows)
    df.to_csv("linreg_epoch_log.csv", index=False)
    coef_df = pd.DataFrame({"feature": FEATURE_NAMES, "theta": model.theta})
    coef_df.to_csv("linreg_theta.csv", index=False)
    print("Saved: linreg_epoch_log.csv, linreg_theta.csv")

if __name__ == "__main__":
    train()
