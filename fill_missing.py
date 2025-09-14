# fix_entsoe_missing.py
import pandas as pd
import numpy as np

RAW_FILE = "dayahead_nl_2024_raw.csv"     # <-- jouw ruwe bestand (NIET missing_hours_2024.csv)
OUT_FILE = "dayahead_nl_2024_filled.csv"
TZ_LOCAL = "Europe/Amsterdam"

def pick_timestamp(df):
    # voorkeursvolgorde
    for c in ["timestamp_utc","timestamp_europe_amsterdam","timestamp_local","mtu_start","time","datetime"]:
        if c in df.columns:
            return c
    raise ValueError("Geen bruikbare timestamp-kolom gevonden.")

def read_prices_raw(path):
    df = pd.read_csv(path)
    tcol = pick_timestamp(df)

    # Tijdstempel -> UTC index
    if tcol == "timestamp_utc":
        ts_utc = pd.to_datetime(df[tcol].astype(str).str.strip(), errors="coerce", utc=True)
    else:
        ts_local = pd.to_datetime(df[tcol].astype(str).str.strip(), errors="coerce")
        if getattr(ts_local.dt, "tz", None) is None:
            ts_local = ts_local.dt.tz_localize(TZ_LOCAL)
        ts_utc = ts_local.dt.tz_convert("UTC")

    # Prijs kolom kiezen en numeric maken
    p = pd.Series(dtype=float, index=np.arange(len(df)))
    if "price_eur_per_mwh" in df.columns:
        p = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce")
    elif "price" in df.columns:
        # vaak EUR/MWh met unit kolom
        raw = pd.to_numeric(df["price"], errors="coerce")
        if "unit" in df.columns and df["unit"].astype(str).str.upper().str.contains("MWH").any():
            p = raw
        else:
            # aannemen dat 'price' EUR/kWh is
            p = raw * 1000.0
    elif "price_eur_per_kwh" in df.columns:
        p = pd.to_numeric(df["price_eur_per_kwh"], errors="coerce") * 1000.0
    else:
        # laatste redmiddel: eerste numerieke kolom
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            p = pd.to_numeric(df[num_cols[0]], errors="coerce")
        else:
            p = pd.Series(dtype=float)

    base = (pd.DataFrame({"price_eur_per_mwh": p.values}, index=ts_utc)
              .dropna(subset=["price_eur_per_mwh"])
              .sort_index())

    if base.empty:
        raise ValueError(
            "Geen geldige prijswaarden gevonden. "
            "Gebruik je wel het RUWE ENTSO-E bestand (niet missing_hours_2024.csv)?"
        )

    # duplicates middelen
    base = base.groupby(level=0)["price_eur_per_mwh"].mean().to_frame()
    return base

def main():
    base = read_prices_raw(RAW_FILE)

    # volledige 2024-range opbouwen (in UTC)
    start_utc = pd.Timestamp("2024-01-01", tz=TZ_LOCAL).tz_convert("UTC") - pd.Timedelta(hours=1)
    end_utc   = pd.Timestamp("2025-01-01", tz=TZ_LOCAL).tz_convert("UTC")
    full_idx  = pd.date_range(start_utc, end_utc, freq="1H", tz="UTC")

    print("Ontbrekende uren vóór vullen:", len(full_idx.difference(base.index)))

    # reindex + ffill + bfill
    filled = base.reindex(full_idx)
    filled["price_eur_per_mwh"] = filled["price_eur_per_mwh"].ffill().bfill()

    # voeg gemaksvelden toe
    filled["price_eur_per_kwh"] = filled["price_eur_per_mwh"] / 1000.0
    filled["timestamp_utc"] = filled.index
    filled["timestamp_europe_amsterdam"] = filled.index.tz_convert(TZ_LOCAL)
    filled["currency"] = "EUR"
    filled["unit"] = "MWH"

    out = filled.reset_index(drop=True)[[
        "timestamp_utc","price_eur_per_mwh","price_eur_per_kwh",
        "currency","unit","timestamp_europe_amsterdam"
    ]]
    out.to_csv(OUT_FILE, index=False)
    print(f"✅ Geschreven: {OUT_FILE}")
    print("Ontbrekende uren ná vullen:", 0)

if __name__ == "__main__":
    main()
