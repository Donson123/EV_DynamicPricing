# shift_sessions_to_2024_weeks.py
# Verschuif sessies 52 weken terug in lokale tijd (Europe/Amsterdam) → behoud weekdag & tijdstip.
# Input: sessions.csv met kolommen: arrival, departure, energy_kWh[, cp_id, pmax_kW/peak_kW]
# Output: sessions_2024.csv

import pandas as pd
import numpy as np

INP = "sessions.csv"
OUT = "sessions_2024.csv"
TZ  = "Europe/Amsterdam"
WEEKS_SHIFT = 52  # exact 52 weken terug

def to_aware_utc(s, tz_local=TZ):
    # Probeer eerst strings met offset direct naar UTC
    ts = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", utc=True)
    if ts.isna().mean() > 0.5:
        # Val terug: parse zonder utc en localize/convert
        ts_local = pd.to_datetime(s.astype(str).str.strip(), errors="coerce")
        if getattr(ts_local.dt, "tz", None) is None:
            ts_local = ts_local.dt.tz_localize(tz_local)
        else:
            ts_local = ts_local.dt.tz_convert(tz_local)
        ts = ts_local.dt.tz_convert("UTC")
    return ts

def main():
    df = pd.read_csv(INP)

    if not {"arrival","departure","energy_kWh"}.issubset(df.columns):
        raise ValueError("sessions.csv mist verplichte kolommen: arrival, departure, energy_kWh")

    arr_utc = to_aware_utc(df["arrival"])
    dep_utc = to_aware_utc(df["departure"])

    # Naar lokale tijd → 52 weken terug → terug naar UTC
    arr_loc = arr_utc.dt.tz_convert(TZ)
    dep_loc = dep_utc.dt.tz_convert(TZ)

    delta = pd.to_timedelta(f"{WEEKS_SHIFT}W")  # exact 52 weken
    arr_loc_shift = arr_loc - delta
    dep_loc_shift = dep_loc - delta

    arr_utc_shift = arr_loc_shift.dt.tz_convert("UTC")
    dep_utc_shift = dep_loc_shift.dt.tz_convert("UTC")

    out = df.copy()
    # Overschrijf de originele arrival/departure met shiftte waarden als ISO (met Z offset)
    out["arrival"]   = arr_utc_shift.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["departure"] = dep_utc_shift.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # pmax_kW veld normaliseren (gebruik peak_kW indien aanwezig)
    if "pmax_kW" not in out.columns:
        if "peak_kW" in out.columns:
            out["pmax_kW"] = pd.to_numeric(out["peak_kW"], errors="coerce").fillna(11.0)
        else:
            out["pmax_kW"] = 11.0
    else:
        out["pmax_kW"] = pd.to_numeric(out["pmax_kW"], errors="coerce").fillna(11.0)

    # sanity: arrival < departure, energie > 0
    out = out.assign(
        _a=pd.to_datetime(out["arrival"], utc=True, errors="coerce"),
        _d=pd.to_datetime(out["departure"], utc=True, errors="coerce"),
        energy_kWh=pd.to_numeric(out["energy_kWh"], errors="coerce")
    )
    before = len(out)
    out = out[(out["_a"].notna()) & (out["_d"].notna()) & (out["_d"] > out["_a"]) & (out["energy_kWh"] > 1e-6)].copy()
    dropped = before - len(out)

    # weg met helperkolommen
    out = out.drop(columns=["_a","_d"])

    out.to_csv(OUT, index=False)
    print(f"✅ Geschreven: {OUT} (dropped {dropped} ongeldige rijen)")
    # snelle diagnose
    print("Voorbeeldregels:")
    print(out.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
