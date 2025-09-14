# get_dayahead_nl_2024_hourly.py
# Haalt ENTSO-E TP day-ahead prices (A44) per uur op voor NL (2024) en schrijft naar CSV.

import os
import sys
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

import pandas as pd
import requests

BASE = "https://web-api.tp.entsoe.eu/api"
BIDDING_ZONE = "10YNL----------L"   # NL bidding zone
DOCUMENT_TYPE = "A44"               # Day-ahead prices
TOKEN = "e2e3c6a3-9053-432b-b6f0-2609f18e04f8"

def parse_dt_utc(s: str) -> datetime:
    # '2024-01-01T00:00Z' or '...+00:00' -> aware UTC
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def parse_iso_duration(d: str) -> timedelta:
    d = d.upper()
    if d in ("PT1H", "PT60M"): return timedelta(hours=1)
    if d == "PT30M": return timedelta(minutes=30)
    if d == "PT15M": return timedelta(minutes=15)
    if d.startswith("PT") and d.endswith("M"): return timedelta(minutes=int(d[2:-1]))
    if d.startswith("PT") and d.endswith("H"): return timedelta(hours=int(d[2:-1]))
    raise ValueError(f"Unsupported resolution: {d}")

def ns_uri(tag: str) -> str:
    return tag[1:].split('}')[0] if tag.startswith("{") else ""

def fetch_xml(period_start: datetime, period_end: datetime) -> str:
    if not TOKEN or TOKEN in ("YOUR_TOKEN", "PLAATS_HIER_JE_TOKEN"):
        print("Fout: zet je token in env var TP_TOKEN of vul het in het script bij TOKEN.")
        sys.exit(1)
    params = {
        "documentType": DOCUMENT_TYPE,
        "in_Domain": BIDDING_ZONE,
        "out_Domain": BIDDING_ZONE,
        "periodStart": period_start.strftime("%Y%m%d%H%M"),
        "periodEnd": period_end.strftime("%Y%m%d%H%M"),
        "securityToken": TOKEN,
    }
    r = requests.get(BASE, params=params, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP {r.status_code} voor {params['periodStart']}–{params['periodEnd']}: {e}")
        print("Response body (ingekort):\n", r.text[:2000])
        sys.exit(1)
    return r.text

def parse_prices(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    uri = ns_uri(root.tag)
    ns = {"ns": uri} if uri else {}
    q = (lambda name: f"ns:{name}") if uri else (lambda name: name)

    rows = []
    for ts in root.findall(f".//{q('TimeSeries')}", ns):
        unit = (ts.findtext(f".//{q('price_Measure_Unit.name')}", default="", namespaces=ns)
                or ts.findtext(f".//{q('measure_Unit.name')}", default="", namespaces=ns))
        currency = ts.findtext(f".//{q('currency_Unit.name')}", default="", namespaces=ns)

        for per in ts.findall(q("Period"), ns):
            res = per.findtext(q("resolution"), namespaces=ns)
            step = parse_iso_duration(res)
            tstart = per.findtext(f"{q('timeInterval')}/{q('start')}", namespaces=ns)
            if not tstart:
                continue
            start_dt = parse_dt_utc(tstart)

            for pt in per.findall(q("Point"), ns):
                pos_txt = pt.findtext(q("position"), namespaces=ns)
                val_txt = pt.findtext(q("price.amount"), namespaces=ns)
                if not pos_txt or val_txt is None:
                    continue
                pos = int(pos_txt)
                ts_utc = start_dt + (pos - 1) * step
                rows.append({
                    "timestamp_utc": ts_utc,
                    "price_eur_per_mwh": float(val_txt),
                    "currency": currency or "EUR",
                    "unit": unit or "EUR/MWh",
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp_utc")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_europe_amsterdam"] = df["timestamp_utc"].dt.tz_convert("Europe/Amsterdam")
    return df

def month_starts_utc(year: int):
    dt = datetime(year, 1, 1, tzinfo=timezone.utc)
    for m in range(1, 13):
        yield dt.replace(month=m, day=1, hour=0, minute=0)
    yield datetime(year + 1, 1, 1, tzinfo=timezone.utc)

def main():
    # Ophalen per maand voor 2024
    dfs = []
    months = list(month_starts_utc(2024))
    for i in range(12):
        start = months[i]
        end = months[i+1]
        xml = fetch_xml(start, end)
        df = parse_prices(xml)
        print(f"{start:%Y-%m}: {len(df)} punten")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if all_df.empty:
        print("Geen data ontvangen.")
        sys.exit(1)

    # Dedup en sorteer
    all_df = (all_df
              .drop_duplicates(subset=["timestamp_utc"])
              .sort_values("timestamp_utc")
              .reset_index(drop=True))

    # Controle op compleetheid (verwacht 8784 uren in 2024, UTC)
    expected = pd.date_range("2024-01-01 00:00", "2025-01-01 00:00",
                             freq="H", tz="UTC", inclusive="left")
    merged = pd.DataFrame({"timestamp_utc": expected}).merge(
        all_df, on="timestamp_utc", how="left"
    )
    missing = merged[merged["price_eur_per_mwh"].isna()]
    if len(missing) > 0:
        missing.to_csv("missing_hours_2024.csv", index=False)
        print(f"Waarschuwing: {len(missing)} ontbrekende uren. Zie missing_hours_2024.csv")

    # Schrijf definitieve CSV
    cols = ["timestamp_utc", "timestamp_europe_amsterdam", "price_eur_per_mwh", "currency", "unit"]
    merged[cols].to_csv("dayahead_nl_2024_hourly.csv", index=False)
    print(f"✅ Geschreven: dayahead_nl_2024_hourly.csv ({len(merged)} rijen)")

if __name__ == "__main__":
    main()
