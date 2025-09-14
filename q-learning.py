# qlearn_split.py
# Vereisten: pip install pandas numpy

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import math
from decimal import Decimal
import copy

# =========== CONFIG ===========
# Verwachte sessie-splits (gemaakt met jouw build_sessions_splits.py)
SESSIONS_TRAIN = "sessions_train.csv"
SESSIONS_VAL   = "sessions_validation.csv"
SESSIONS_TEST  = "sessions_test.csv"

# Fallback (als splits ontbreken)
SESSIONS_FALLBACK = "sessions_2024.csv"

PRICES_CSV   = "dayahead_nl_2024_filled.csv"
TZ_LOCAL     = "Europe/Amsterdam"

# Gebruik het hele jaar; dagen zonder sessies geven 0 reward (oké)
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

# Actie-ruimte: 0.10 t/m 1.00 in stappen van 0.01
ACTION_MIN_EUR  = Decimal("0.10")
ACTION_MAX_EUR  = Decimal("1.00")
ACTION_STEP_EUR = Decimal("0.01")

DT_MIN = 15
DT_H   = DT_MIN/60.0

SITE_PMAX_KW = 11.0 * 1

# Economie / doelen
ALPHA = 2.0
RETAIL_ACTIONS = [float(ACTION_MIN_EUR + i*ACTION_STEP_EUR)
                  for i in range(int((ACTION_MAX_EUR - ACTION_MIN_EUR)/ACTION_STEP_EUR) + 1)]
REF_PRICE = 0.35
ELASTICITY = 1.5
URG_ETA = 0.25

# Q-learning
GAMMA = 0.98
ALPHA_Q = 0.3
EPSILON_START = 0.2
EPSILON_END   = 0.02
EPSILON_DECAY_DAYS = 60
EPOCHS = 30

RNG_SEED = 42
# ==============================

rng = np.random.default_rng(RNG_SEED)
PRICE_QS = None  # wordt lazy gevuld

# ---------- Helpers ----------
def _read_sessions_csv(path: str) -> pd.DataFrame:
    s = pd.read_csv(path)
    for c in ("arrival","departure","energy_kWh"):
        if c not in s.columns:
            raise ValueError(f"{path} mist kolom '{c}'")
    arr_utc = pd.to_datetime(s["arrival"].astype(str).str.strip(), errors="coerce", utc=True)
    dep_utc = pd.to_datetime(s["departure"].astype(str).str.strip(), errors="coerce", utc=True)
    if arr_utc.isna().mean() > 0.5 or dep_utc.isna().mean() > 0.5:
        arr_local = pd.to_datetime(s["arrival"].astype(str).str.strip(), errors="coerce", utc=False)
        dep_local = pd.to_datetime(s["departure"].astype(str).str.strip(), errors="coerce", utc=False)
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
    s = s.dropna(subset=["arrival_utc","departure_utc","energy_kWh"])
    s = s[s["energy_kWh"] > 1e-6].copy()
    if "cp_id" not in s.columns:
        s["cp_id"] = "cp0"

    cols = ["cp_id","arrival_utc","departure_utc","energy_kWh","pmax_kW"]
    return s[cols].reset_index(drop=True)

def load_sessions_splits():
    p_train = Path(SESSIONS_TRAIN)
    p_val   = Path(SESSIONS_VAL)
    p_test  = Path(SESSIONS_TEST)
    if p_train.exists() and p_val.exists() and p_test.exists():
        print("✅ Splits gevonden: sessions_train/validation/test.csv")
        return (_read_sessions_csv(str(p_train)),
                _read_sessions_csv(str(p_val)),
                _read_sessions_csv(str(p_test)))
    # fallback: één bestand voor alle splits
    p_fb = Path(SESSIONS_FALLBACK)
    if not p_fb.exists():
        raise FileNotFoundError("Geen sessie-splits en geen fallback sessions_2024.csv gevonden.")
    print("⚠️  Splits niet gevonden; gebruik fallback voor alle splits:", p_fb)
    df = _read_sessions_csv(str(p_fb))
    return df.copy(), df.copy(), df.copy()

def load_prices(path=PRICES_CSV) -> pd.Series:
    df = pd.read_csv(path)

    # tijdkolom kiezen
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
        raise ValueError("Kon geen tijdkolom vinden in prijzen CSV.")

    # prijs naar EUR/kWh
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

def willingness(price):
    x = (REF_PRICE - price) * ELASTICITY
    return 1.0/(1.0 + np.exp(-x))

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
        self.pr = price_series_utc.copy()  # UTC-index Series
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
        global PRICE_QS
        slot = int(((self.t_utc.tz_convert(TZ_LOCAL) - self.t_utc.tz_convert(TZ_LOCAL).normalize()).total_seconds() // (DT_MIN*60)) % (int(24*60/DT_MIN)))
        mask = (self.jobs["arrival_utc"] <= self.t_utc) & (self.jobs["departure_utc"] > self.t_utc) & (self.jobs["energy_left_kWh"] > 1e-9)
        n_act = int(mask.sum()); n_act_bin = min(n_act, 3)
        w = self._price_at(self.t_utc)
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

# ---------- Q-learning ----------
class QTable:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))

    def policy(self, state, epsilon):
        if rng.random() < epsilon:
            return rng.integers(self.n_actions)
        q = self.Q[state]
        return int(np.argmax(q + 1e-9*rng.normal(size=self.n_actions)))  # tie-break

    def greedy_action(self, state):
        q = self.Q[state]
        return int(np.argmax(q))

    def update(self, s, a, r, s_next, alpha=ALPHA_Q, gamma=GAMMA):
        qsa = self.Q[s][a]
        max_next = np.max(self.Q[s_next])
        self.Q[s][a] = (1 - alpha)*qsa + alpha*(r + gamma*max_next)

# ---------- Loops ----------
def run_epoch_train(env: PricingEnv, qtab: QTable, epoch_idx: int):
    """Train op alle dagen (ε-schedule); retourneert gemiddelden."""
    day_idx = 0
    sum_loss = sum_rev = sum_peak = 0.0
    n_days = 0

    for day_local in daterange_local_days(START_DATE, END_DATE):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-day_idx / max(1, EPSILON_DECAY_DAYS))
        day_idx += 1

        s = env.reset(day_local)
        done = False
        day_rev = 0.0
        day_peak = 0.0
        while not done:
            a_idx  = qtab.policy(s, epsilon)
            retail = RETAIL_ACTIONS[a_idx]
            s2, reward, done, info = env.step(retail)
            qtab.update(s, a_idx, reward, s2)
            s = s2
            day_rev  += info.get("revenue", 0.0)
            day_peak += info.get("peak_kWh", 0.0)

        day_loss = -day_rev + ALPHA * day_peak
        sum_loss += day_loss
        sum_rev  += day_rev
        sum_peak += day_peak
        n_days   += 1

    return (sum_loss/n_days, sum_rev/n_days, sum_peak/n_days)

def run_epoch_eval(env: PricingEnv, qtab: QTable):
    """Evaluatie greedy (ε=0) op alle dagen; retourneert gemiddelden."""
    sum_loss = sum_rev = sum_peak = 0.0
    n_days = 0
    for day_local in daterange_local_days(START_DATE, END_DATE):
        s = env.reset(day_local)
        done = False
        day_rev = 0.0
        day_peak = 0.0
        while not done:
            a_idx = qtab.greedy_action(s)
            retail = RETAIL_ACTIONS[a_idx]
            s, reward, done, info = env.step(retail)
            day_rev  += info.get("revenue", 0.0)
            day_peak += info.get("peak_kWh", 0.0)
        day_loss = -day_rev + ALPHA * day_peak
        sum_loss += day_loss
        sum_rev  += day_rev
        sum_peak += day_peak
        n_days   += 1
    return (sum_loss/n_days, sum_rev/n_days, sum_peak/n_days)

def train():
    # Data inladen
    sessions_train, sessions_val, sessions_test = load_sessions_splits()
    prices = load_prices(PRICES_CSV)

    # Envs per split
    env_train = PricingEnv(sessions_train, prices, site_pmax=SITE_PMAX_KW)
    env_val   = PricingEnv(sessions_val,   prices, site_pmax=SITE_PMAX_KW)
    env_test  = PricingEnv(sessions_test,  prices, site_pmax=SITE_PMAX_KW)

    # Q-table
    qtab = QTable(n_actions=len(RETAIL_ACTIONS))

    # Logs
    rows_train = []   # (epoch, avg_loss, avg_revenue, avg_peak_kWh)
    rows_val   = []   # (epoch, avg_loss, avg_revenue, avg_peak_kWh)

    best_val_loss = float("inf")
    best_epoch = None
    best_Q = None

    # Reset globale prijs-kwartielen één keer
    global PRICE_QS
    PRICE_QS = None

    for epoch in range(1, EPOCHS+1):
        print(f"\n=== EPOCH {epoch}/{EPOCHS} ===")

        # TRAIN (update Q)
        avg_loss_tr, avg_rev_tr, avg_peak_tr = run_epoch_train(env_train, qtab, epoch)
        rows_train.append((epoch, avg_loss_tr, avg_rev_tr, avg_peak_tr))

        # VALIDATION (geen updates)
        avg_loss_val, avg_rev_val, avg_peak_val = run_epoch_eval(env_val, qtab)
        rows_val.append((epoch, avg_loss_val, avg_rev_val, avg_peak_val))

        print(f"Train: avg_loss={avg_loss_tr:.4f}  avg_rev={avg_rev_tr:.2f}  avg_peak={avg_peak_tr:.2f}")
        print(f"Valid: avg_loss={avg_loss_val:.4f}  avg_rev={avg_rev_val:.2f}  avg_peak={avg_peak_val:.2f}")

        # Beste policy bijhouden op validation loss
        if avg_loss_val < best_val_loss:
            best_val_loss = avg_loss_val
            best_epoch = epoch
            best_Q = copy.deepcopy(qtab.Q)

    # Schrijf epoch logs
    df_tr  = pd.DataFrame(rows_train, columns=["epoch","avg_loss","avg_revenue","avg_peak_kWh"])
    df_val = pd.DataFrame(rows_val,   columns=["epoch","avg_loss","avg_revenue","avg_peak_kWh"])
    df_tr.to_csv("qlearn_epoch_log_train.csv", index=False)
    df_val.to_csv("qlearn_epoch_log_validation.csv", index=False)
    print("💾 Logs geschreven: qlearn_epoch_log_train.csv, qlearn_epoch_log_validation.csv")

    # TEST met beste epoch
    if best_Q is None:
        print("⚠️ Geen beste epoch gevonden? Valideer input/duur.")
        return
    qtab_best = QTable(n_actions=len(RETAIL_ACTIONS))
    qtab_best.Q = best_Q
    avg_loss_te, avg_rev_te, avg_peak_te = run_epoch_eval(env_test, qtab_best)

    pd.DataFrame([{
        "best_epoch": best_epoch,
        "val_best_loss": round(best_val_loss, 6),
        "test_avg_loss": round(avg_loss_te, 6),
        "test_avg_revenue": round(avg_rev_te, 6),
        "test_avg_peak_kWh": round(avg_peak_te, 6),
    }]).to_csv("qlearn_test_summary.csv", index=False)
    print(f"🏁 Beste epoch: {best_epoch} | test avg_loss={avg_loss_te:.4f} avg_rev={avg_rev_te:.2f} avg_peak={avg_peak_te:.2f}")
    print("💾 Test samenvatting: qlearn_test_summary.csv")

    # Policy dump (beste)
    with open("qlearn_policy_dump_best.txt","w", encoding="utf-8") as f:
        for s, qs in best_Q.items():
            f.write(f"{s}\t{list(qs)}\n")
    print("💾 Beste policy dump: qlearn_policy_dump_best.txt")

if __name__ == "__main__":
    train()
