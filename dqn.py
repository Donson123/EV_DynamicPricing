# dqn_dynamic_pricing.py
# Vereisten: pip install pandas numpy torch

import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

# =========== CONFIG ===========
SESSIONS_CSV = "sessions_2024.csv"               # uit build_sessions.py (arrival/departure/energy_kWh)
PRICES_CSV   = "dayahead_nl_2024_filled.csv"     # ENTSO-E day-ahead (mag met lokale of UTC timestamps)
TZ_LOCAL     = "Europe/Amsterdam"

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

DT_MIN = 15
DT_H   = DT_MIN/60.0

SITE_PMAX_KW = 11.0 * 1  # totale sitescap (kW)

# Economie / doelen
ALPHA = 2.5
RETAIL_ACTIONS = [0.15, 0.25, 0.35, 0.45, 0.60]  # €/kWh
REF_PRICE = 0.35
ELASTICITY = 1.5
URG_ETA = 0.25

# DQN / training
EPOCHS = 100                   # aantal keer dat je het hele jaar doorloopt
GAMMA = 0.98
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000
WARMUP_STEPS = 2_000           # pas leren na voldoende buffer
TARGET_UPDATE_EVERY = 2_000    # hard update in stappen
GRAD_CLIP = 5.0
HUBER_DELTA = 1.0              # SmoothL1Loss delta (PyTorch standaard)

# epsilon-decay per epoch (dagelijks berekend)
EPSILON_START = 0.2
EPSILON_END   = 0.02
EPSILON_DECAY_DAYS = 60

RNG_SEED = 42
# ==============================

rng = np.random.default_rng(RNG_SEED)
torch.manual_seed(RNG_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Helpers: inladen ----------
def load_sessions(path=SESSIONS_CSV):
    s = pd.read_csv(path)

    for c in ("arrival", "departure", "energy_kWh"):
        if c not in s.columns:
            raise ValueError(f"sessions.csv mist kolom '{c}'")

    arr_utc = pd.to_datetime(s["arrival"].astype(str).str.strip(), errors="coerce", utc=True)
    dep_utc = pd.to_datetime(s["departure"].astype(str).str.strip(), errors="coerce", utc=True)

    # fallback naar lokale tijd
    if arr_utc.isna().mean() > 0.5 or dep_utc.isna().mean() > 0.5:
        arr_local = pd.to_datetime(s["arrival"].astype(str).str.strip(), errors="coerce")
        dep_local = pd.to_datetime(s["departure"].astype(str).str.strip(), errors="coerce")
        if getattr(arr_local.dt, "tz", None) is None:
            arr_local = arr_local.dt.tz_localize(TZ_LOCAL)
        if getattr(dep_local.dt, "tz", None) is None:
            dep_local = dep_local.dt.tz_localize(TZ_LOCAL)
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

    return s[["cp_id","arrival_utc","departure_utc","energy_kWh","pmax_kW"]].reset_index(drop=True)

def load_prices(path=PRICES_CSV):
    df = pd.read_csv(path)

    # 1) timestamps → UTC-index
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
        raise ValueError("Kon geen tijdkolom vinden (timestamp_europe_amsterdam / timestamp_local / timestamp_utc).")

    # 2) prijs → EUR/kWh
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

    best, best_nonnull = None, -1
    for series in cand.values():
        nn = int(series.notna().sum())
        if nn > best_nonnull:
            best, best_nonnull = series, nn
    if best is None or best_nonnull == 0:
        raise ValueError("Geen bruikbare prijzen gevonden.")

    prices = (pd.DataFrame({"eur_per_kwh": best.values}, index=ts_utc)
                .dropna(subset=["eur_per_kwh"])
                .sort_index())

    prices = prices.groupby(level=0)["eur_per_kwh"].mean().to_frame()
    pr15 = prices["eur_per_kwh"].resample("15min").ffill()

    # volledige window afdwingen
    start_utc = pd.Timestamp(START_DATE, tz=TZ_LOCAL).tz_convert("UTC") - pd.Timedelta(hours=1)
    end_utc   = pd.Timestamp(END_DATE,   tz=TZ_LOCAL).tz_convert("UTC")
    full_idx  = pd.date_range(start_utc, end_utc, freq="15min", tz="UTC")
    pr15 = pr15.reindex(full_idx).ffill().bfill()

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

# ---------- Vraagmodel ----------
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
PRICE_QS = None

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

    # Discrete state -> tuple (slot, n_act_bin, wbin, urg_bin)
    def _state(self):
        slot = int(((self.t_utc.tz_convert(TZ_LOCAL) - self.t_utc.tz_convert(TZ_LOCAL).normalize()).total_seconds()
                    // (DT_MIN*60)) % (int(24*60/DT_MIN)))
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

# ---------- State encoding ----------
# One-hot voor elk deel: slot(96) + n_act_bin(4) + wbin(4) + urg_bin(4) = 108
N_SLOT = int(24*60/DT_MIN)  # 96
def encode_state(s):
    slot, n_act_bin, wbin, urg_bin = s
    v = np.zeros(N_SLOT + 4 + 4 + 4, dtype=np.float32)
    v[slot] = 1.0
    v[N_SLOT + n_act_bin] = 1.0
    v[N_SLOT + 4 + wbin] = 1.0
    v[N_SLOT + 8 + urg_bin] = 1.0
    return v

# ---------- Replay buffer ----------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.states = np.zeros((capacity, N_SLOT+12), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, N_SLOT+12), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, s2, done):
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        size = self.capacity if self.full else self.pos
        idx = rng.integers(0, size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.capacity if self.full else self.pos

# ---------- DQN netwerk ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128,128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Policy helper ----------
def epsilon_greedy(q_values, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[-1]))
    return int(torch.argmax(q_values).item())

# ---------- Training ----------
def train():
    sessions = load_sessions(SESSIONS_CSV)
    prices   = load_prices(PRICES_CSV)
    env = PricingEnv(sessions, prices, site_pmax=SITE_PMAX_KW)

    n_actions = len(RETAIL_ACTIONS)
    online = MLP(N_SLOT+12, n_actions).to(device)
    target = MLP(N_SLOT+12, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    opt = optim.Adam(online.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss(beta=HUBER_DELTA)  # Huber

    rb = ReplayBuffer(REPLAY_CAPACITY)

    # Logging per epoch
    epoch_rows = []  # (epoch, avg_loss, avg_revenue, avg_peak_kWh, avg_td_loss)

    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        sum_rev = sum_peak = 0.0
        sum_business_loss = 0.0
        sum_td_loss = 0.0
        n_days = 0
        n_td = 0

        for day_local in daterange_local_days(START_DATE, END_DATE):
            # epsilon schema per dag
            day_idx = (day_local - pd.Timestamp(START_DATE, tz=TZ_LOCAL)).days
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-day_idx / max(1, EPSILON_DECAY_DAYS))

            s = env.reset(day_local)
            done = False
            day_rev = 0.0
            day_peak = 0.0

            # Encode eerste state
            s_vec = encode_state(s)

            while not done:
                # actie kiezen
                with torch.no_grad():
                    q = online(torch.from_numpy(s_vec).unsqueeze(0).to(device))
                a_idx = epsilon_greedy(q, epsilon)
                retail = RETAIL_ACTIONS[a_idx]

                # environment stap
                s2, reward, done, info = env.step(retail)
                s2_vec = encode_state(s2)

                # buffer
                rb.push(s_vec, a_idx, reward, s2_vec, done)

                # boekhouding
                day_rev  += info.get("revenue", 0.0)
                day_peak += info.get("peak_kWh", 0.0)

                # learn step
                if len(rb) >= WARMUP_STEPS:
                    states, actions, rewards, next_states, dones = rb.sample(BATCH_SIZE)

                    states_t = torch.from_numpy(states).to(device)
                    actions_t = torch.from_numpy(actions).unsqueeze(1).to(device)
                    rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(device)
                    next_states_t = torch.from_numpy(next_states).to(device)
                    dones_t = torch.from_numpy(dones).unsqueeze(1).to(device)

                    # Q(s,a)
                    q_sa = online(states_t).gather(1, actions_t)

                    # target: r + γ * max_a' Q_target(s',a') * (1 - done)
                    with torch.no_grad():
                        q_next = target(next_states_t).max(dim=1, keepdim=True)[0]
                        target_q = rewards_t + GAMMA * (1.0 - dones_t) * q_next

                    loss = criterion(q_sa, target_q)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
                    opt.step()

                    sum_td_loss += float(loss.item())
                    n_td += 1

                    # target update (hard)
                    if (global_step % TARGET_UPDATE_EVERY) == 0:
                        target.load_state_dict(online.state_dict())

                global_step += 1
                s_vec = s2_vec

            # dag klaar
            sum_rev  += day_rev
            sum_peak += day_peak
            n_days   += 1

            # jouw business loss per dag
            day_business_loss = -day_rev + ALPHA * day_peak
            sum_business_loss += day_business_loss

        avg_loss    = sum_business_loss / max(1, n_days)     # jouw metric
        avg_rev     = sum_rev / max(1, n_days)
        avg_peak    = sum_peak / max(1, n_days)
        avg_td_loss = (sum_td_loss / max(1, n_td)) if n_td else float("nan")

        epoch_rows.append((epoch+1, avg_loss, avg_rev, avg_peak, avg_td_loss))
        print(f"== EPOCH {epoch+1} klaar ==  "
              f"avg_loss={avg_loss:.4f}  avg_rev={avg_rev:.2f}  "
              f"avg_peak_kWh={avg_peak:.2f}  avg_td_loss={avg_td_loss:.4f}")

    # CSV loggen
    df = pd.DataFrame(epoch_rows,
                      columns=["epoch","avg_loss","avg_revenue","avg_peak_kWh","avg_td_loss"])
    df.to_csv("dqn_epoch_log.csv", index=False)
    print("\nLog → dqn_epoch_log.csv")
    print(df.tail())

if __name__ == "__main__":
    train()
