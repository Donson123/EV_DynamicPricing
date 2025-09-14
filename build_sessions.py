# build_sessions.py — random train/val/test splits met lokale (Europe/Amsterdam) tijden
import pandas as pd
from pathlib import Path
import glob, re
import numpy as np

# ===== Instellingen =====
INPUT_GLOB = "input/profile_*.csv"   # pas aan indien nodig
PREFERRED_POWER_COL = "profile_0_1"  # kolom met het CP-vermogen
POWER_THRESHOLD = 0.01               # kW > 0.01 = laden
GAP_TOLERANCE_STEPS = 0              # 0=uit; 1–2 laat korte 0-gaten binnen sessie toe
LOCAL_TZ = "Europe/Amsterdam"

# Random split-config (altijd random)
SPLIT_SCOPE = "cp"        # "cp" = split per laadpunt (cp_id), "session" = split individuele sessies
RANDOM_RATIOS = (0.7, 0.15, 0.15)  # train, val, test — moet optellen tot 1.0
RANDOM_SEED   = 42

# Uitvoer
OUT_ALL  = "sessions_all.csv"
OUT_TRAIN = "sessions_train.csv"
OUT_VAL   = "sessions_validation.csv"
OUT_TEST  = "sessions_test.csv"


def find_time_col(cols):
    cols = [c.strip() for c in cols]
    # voorkeursnamen
    for cand in ("datetime","timestamp_local","timestamp","time","DateTime","DATE","TIME"):
        for c in cols:
            if c.lower() == cand.lower():
                return c
    # fallback: iets met time/date
    for c in cols:
        if re.search(r"time|date", c.lower()):
            return c
    return None

def pick_power_col(cols):
    if PREFERRED_POWER_COL in cols:
        return PREFERRED_POWER_COL
    for c in cols:
        if re.search(r"(?:^|_)0_1$", c):  # bv. profile_0_1
            return c
    tcol = find_time_col(cols)
    others = [c for c in cols if c != tcol]
    return others[-1] if others else None

def parse_to_local(series_str: pd.Series) -> pd.Series:
    """
    1) Probeer te parsen met utc=True (pakt ISO met +01/+02/+00 netjes op).
    2) Converteer naar Europe/Amsterdam.
    3) Voor entries die dan nog NaT zijn: parse zonder utc, en localize naar Europe/Amsterdam.
    """
    ts_utc = pd.to_datetime(series_str.astype(str).str.strip(), errors="coerce", utc=True)
    ts_local = ts_utc.dt.tz_convert(LOCAL_TZ)

    # entries die niet als UTC parse-baar waren (NaT) opnieuw proberen als lokale tijden
    mask_nat = ts_local.isna()
    if mask_nat.any():
        ts_local_fallback = pd.to_datetime(series_str[mask_nat].astype(str).str.strip(), errors="coerce")
        # als nog tz-loos, localize; als al tz-aware, convert
        if getattr(ts_local_fallback.dt, "tz", None) is None:
            ts_local_fallback = ts_local_fallback.dt.tz_localize(LOCAL_TZ)
        else:
            ts_local_fallback = ts_local_fallback.dt.tz_convert(LOCAL_TZ)
        ts_local.loc[mask_nat] = ts_local_fallback
    return ts_local

def read_profile(path: Path) -> pd.DataFrame:
    # alles als string; sep autodetect; BOM-strip
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    tcol = find_time_col(df.columns)
    if not tcol:
        raise ValueError(f"{path.name}: geen tijdkolom gevonden (headers: {list(df.columns)})")

    ts_local = parse_to_local(df[tcol])

    base = (pd.DataFrame({"datetime": ts_local})
            .dropna()
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"])
            .reset_index(drop=True))

    pcol = pick_power_col(df.columns)
    if not pcol:
        raise ValueError(f"{path.name}: geen vermogen-kolom gevonden")

    # Let op: we nemen dezelfde rij-index als 'base' nu heeft na reset_index
    power = pd.to_numeric(df.loc[base.index, pcol], errors="coerce").fillna(0.0)
    base["kW"] = power.values
    return base

def interval_hours(ts: pd.Series) -> pd.Series:
    # tz-aware; werkt netjes over DST
    dt = ts.shift(-1) - ts
    dt_hours = dt.dt.total_seconds() / 3600.0
    med = dt_hours.dropna().median()
    if pd.isna(med) or med <= 0:
        med = 0.25  # fallback 15 min
    return dt_hours.fillna(med)

def close_small_gaps(on: pd.Series, tol_steps: int) -> pd.Series:
    if tol_steps <= 0:
        return on
    rid = (on != on.shift(fill_value=False)).cumsum()
    df = pd.DataFrame({"val": on, "rid": rid})
    run_len = df.groupby("rid")["val"].transform("size")
    run_val = df.groupby("rid")["val"].transform("first")
    prev_rid = df["rid"].where(run_val.ne(run_val.shift(fill_value=run_val.iloc[0]))).ffill().astype(int)
    next_rid = df["rid"].where(run_val.ne(run_val.shift(-1, fill_value=run_val.iloc[-1]))).bfill().astype(int)
    prev_val = df.groupby(prev_rid)["val"].transform("first")
    next_val = df.groupby(next_rid)["val"].transform("first")
    to_fill = (~df["val"]) & (run_len <= tol_steps) & prev_val & next_val
    out = df["val"].copy()
    out[to_fill] = True
    return out

def extract_sessions_from_file(path: Path):
    df = read_profile(path)
    df["interval_h"] = interval_hours(df["datetime"])
    on = close_small_gaps(df["kW"] > POWER_THRESHOLD, GAP_TOLERANCE_STEPS)
    rid = (on != on.shift(fill_value=False)).cumsum()

    sessions = []
    for _, block in df.assign(on=on, rid=rid).groupby("rid"):
        if not block["on"].iloc[0]:  # 'uit'-run overslaan
            continue
        start = block["datetime"].iloc[0]  # tz-aware Europe/Amsterdam
        end = block["datetime"].iloc[-1] + pd.to_timedelta(block["interval_h"].iloc[-1], unit="h")
        energy_kWh = float((block["kW"] * block["interval_h"]).sum())
        duration_h = float(block["interval_h"].sum())
        sessions.append({
            "source_file": path.name,
            "cp_id": path.stem,        # bv. profile_1
            "arrival": start,          # Europe/Amsterdam (CET/CEST)
            "departure": end,          # Europe/Amsterdam (CET/CEST)
            "duration_min": duration_h * 60.0,
            "energy_kWh": energy_kWh,
            "avg_kW": (energy_kWh / duration_h) if duration_h > 0 else 0.0,
            "peak_kW": float(block["kW"].max()),
            "n_steps": int(len(block)),
        })
    return sessions

# ---------- Splits ----------
def split_random_sessions(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15), seed=42):
    """Random split op sessie-niveau (elk record is 1 sessie)."""
    assert abs(sum(ratios) - 1.0) < 1e-9, "Ratios moeten optellen tot 1.0"
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(ratios[0] * n)
    n_val   = int(ratios[1] * n)
    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train:n_train + n_val].copy()
    test  = df.iloc[n_train + n_val:].copy()
    return train, val, test

def split_random_by_cp(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15), seed=42):
    """Random split op CP-niveau: elke cp_id ligt exclusief in één split."""
    assert abs(sum(ratios) - 1.0) < 1e-9, "Ratios moeten optellen tot 1.0"
    rng = np.random.default_rng(seed)
    cps = df["cp_id"].dropna().unique()
    rng.shuffle(cps)
    n = len(cps)
    n_train = int(ratios[0] * n)
    n_val   = int(ratios[1] * n)
    cp_train = set(cps[:n_train])
    cp_val   = set(cps[n_train:n_train + n_val])
    cp_test  = set(cps[n_train + n_val:])

    train = df[df["cp_id"].isin(cp_train)].copy()
    val   = df[df["cp_id"].isin(cp_val)].copy()
    test  = df[df["cp_id"].isin(cp_test)].copy()
    return train, val, test

def print_summary(name: str, df: pd.DataFrame):
    if df.empty:
        print(f"{name:<10}: 0 sessies")
        return
    total_e = df["energy_kWh"].sum()
    n_cp    = df["cp_id"].nunique()
    dur_h   = df["duration_min"].sum() / 60.0
    print(f"{name:<10}: {len(df):>5} sessies | {n_cp} CP's | {total_e:.1f} kWh | {dur_h:.1f} h totaal")

# ---------- Main ----------
def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        print(f"Geen files gevonden voor patroon: {INPUT_GLOB}")
        return

    all_rows = []
    for f in files:
        try:
            rows = extract_sessions_from_file(Path(f))
            all_rows.extend(rows)
            print(f"✔ {f}: {len(rows)} sessies")
        except Exception as e:
            print(f"⚠ {f} overgeslagen: {e}")

    if not all_rows:
        print("Geen sessies gevonden.")
        return

    out = pd.DataFrame(all_rows).sort_values(["cp_id","arrival"]).reset_index(drop=True)

    # Zorg dat tijdkolommen tz-aware zijn (zou al zo moeten zijn)
    out["arrival"]   = pd.to_datetime(out["arrival"]).dt.tz_convert(LOCAL_TZ)
    out["departure"] = pd.to_datetime(out["departure"]).dt.tz_convert(LOCAL_TZ)

    # Schrijf volledige set (handig voor inspectie)
    out.to_csv(OUT_ALL, index=False)

    # Random splits
    if SPLIT_SCOPE.lower() == "cp":
        train, val, test = split_random_by_cp(out, RANDOM_RATIOS, RANDOM_SEED)
    elif SPLIT_SCOPE.lower() == "session":
        train, val, test = split_random_sessions(out, RANDOM_RATIOS, RANDOM_SEED)
    else:
        raise ValueError("SPLIT_SCOPE moet 'cp' of 'session' zijn.")

    # Schrijven
    train.to_csv(OUT_TRAIN, index=False)
    val.to_csv(OUT_VAL, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("\n✅ Geschreven:")
    print_summary("train", train)
    print_summary("validation", val)
    print_summary("test", test)
    print(f"\nAlle sessies → {OUT_ALL}")
    print(f"Splits → {OUT_TRAIN}, {OUT_VAL}, {OUT_TEST}")

if __name__ == "__main__":
    main()
