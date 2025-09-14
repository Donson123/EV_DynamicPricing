# API-elaad.py (DST-proof v3: fast-start + warmup + short timeouts)
import os, sys, json, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

API_BASE = os.environ.get("SIM_API_BASE", "https://platform.elaad.io/chargingprofilesapi").rstrip("/")
ENDPOINT = "/profile/simulate"
URL = f"{API_BASE}{ENDPOINT}"

BEARER = os.environ.get("SIM_BEARER_TOKEN", "").strip()
API_KEY = os.environ.get("SIM_API_KEY", "").strip()

YEAR = int(os.environ.get("SIM_YEAR", "2025"))
TZ_LOCAL = ZoneInfo("Europe/Amsterdam")
TZ_UTC   = ZoneInfo("UTC")

# Payload-tz: normaal 'CET'; zet SIM_USE_UTC_PAYLOAD=1 voor UTC payload
USE_UTC_PAYLOAD = os.environ.get("SIM_USE_UTC_PAYLOAD", "0") == "1"
TIMEZONE_PARAM  = "UTC" if USE_UTC_PAYLOAD else "CET"

# Tuning via env vars
N_PROFILES = int(os.environ.get("SIM_N_PROFILES", "100"))      # probeer eerst 10 als het zwaar is
INITIAL_CHUNK_MIN = int(os.environ.get("SIM_INIT_CHUNK_MIN", "60"))  # start 60 min
MIN_CHUNK_MIN     = int(os.environ.get("SIM_MIN_CHUNK_MIN", "15"))   # val terug tot 15 min
MAX_CHUNK_HOURS   = int(os.environ.get("SIM_MAX_CHUNK_HOURS", "24"))
REQUEST_TIMEOUT_S = int(os.environ.get("SIM_TIMEOUT_S", "20"))       # korte timeouts forceren snelle fallback
BETWEEN_CALLS_SLEEP_S = float(os.environ.get("SIM_SLEEP_S", "0.2"))
DO_WARMUP = os.environ.get("SIM_WARMUP", "1") == "1"                 # doe 2 warm-ups van 15 min

PROFILE_TYPE = "cp"
LOCATION_TYPE = "public"
VEHICLE_TYPES = os.environ.get("SIM_VEHICLE_TYPES", "car")

OUT_WIDE = f"public_{YEAR}_hourly_wide.csv"
OUT_LONG = f"public_{YEAR}_hourly_long.csv"

def make_headers():
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if BEARER: h["Authorization"] = f"Bearer {BEARER}"
    if API_KEY: h["X-API-Key"] = API_KEY
    return h

def session_with_retries():
    retry = Retry(
        total=4, connect=4, read=4,
        backoff_factor=1.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"POST"}
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def utc_to_local_iso(dt_utc: datetime) -> str:
    return dt_utc.astimezone(TZ_LOCAL).isoformat()

def is_transient(msg: str) -> bool:
    m = (msg or "").lower()
    return any(x in m for x in [
        "http 500","http 502","http 503","http 504","http 429",
        "max retries exceeded","read timed out","timeout","temporarily unavailable"
    ])

def post_simulation(sess: requests.Session, start_iso: str, stop_iso: str) -> dict:
    payload = {
        "start_datetime": start_iso,
        "stop_datetime":  stop_iso,
        "profile_type":   PROFILE_TYPE,
        "n_profiles":     N_PROFILES,
        "vehicle_types":  VEHICLE_TYPES,
        "location_type":  LOCATION_TYPE,
        "timezone":       TIMEZONE_PARAM,
    }
    try:
        r = sess.post(URL, headers=make_headers(), data=json.dumps(payload), timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        # vang timeouts/connection errors als transient
        raise RuntimeError(f"RequestException: {e}")
    try:
        return r.json()
    except ValueError:
        # geen geldige JSON → behandel als transient
        raise RuntimeError(f"Non-JSON response ({r.status_code})")

def to_dataframe(resp: dict) -> pd.DataFrame:
    prof = resp.get("profile", {})
    cp_ids = prof.get("cp_ids", [])
    datetimes_utc = pd.to_datetime(prof.get("datetimes", []), utc=True)
    demands = prof.get("demands_kw", [])

    if not cp_ids or len(datetimes_utc) == 0 or len(demands) == 0:
        return pd.DataFrame()

    T, P = len(datetimes_utc), len(cp_ids)
    if len(demands) == T and all(len(row) == P for row in demands):
        df = pd.DataFrame(demands, index=datetimes_utc, columns=cp_ids)
    elif len(demands) == P and all(len(row) == T for row in demands):
        df = pd.DataFrame(demands, index=cp_ids, columns=datetimes_utc).T
    else:
        raise ValueError(f"Unexpected demands shape: outer={len(demands)}, "
                         f"inner_lengths={{ {set(len(r) for r in demands)} }}, T={T}, P={P}")

    df.index.name = "timestamp_utc"
    df = df.sort_index()
    df["timestamp_local"] = df.index.tz_convert(TZ_LOCAL)
    cols = ["timestamp_local"] + list(df.columns.drop("timestamp_local"))
    return df.reset_index()[cols]

# --- DST-transities in UTC en harde splits ---
def detect_tz_transitions_utc(year: int) -> list[datetime]:
    trans = []
    start = datetime(year, 1, 1, tzinfo=TZ_UTC)
    end   = datetime(year+1, 1, 1, tzinfo=TZ_UTC)

    def offset_at(dt_utc):
        return TZ_LOCAL.utcoffset(dt_utc.astimezone(TZ_LOCAL))

    step = timedelta(hours=1)
    t = start; prev = offset_at(t)
    while t <= end:
        cur = offset_at(t)
        if cur != prev:
            lo, hi = t - step, t
            for _ in range(20):
                mid = lo + (hi - lo)/2
                if offset_at(mid) == prev: lo = mid
                else: hi = mid
            trans.append(hi.astimezone(TZ_UTC))
            prev = cur
        t += step
    return sorted([dt for dt in trans if start <= dt < end])

def warmup(sess, year_start_utc):
    # twee snelle 15m-calls om server te ‘warmen’
    if not DO_WARMUP: return
    for k in range(2):
        s_utc = year_start_utc + timedelta(minutes=15*k)
        e_utc = s_utc + timedelta(minutes=15)
        if USE_UTC_PAYLOAD:
            s_iso, e_iso = s_utc.isoformat(), e_utc.isoformat()
        else:
            s_iso, e_iso = utc_to_local_iso(s_utc), utc_to_local_iso(e_utc)
        try:
            print(f"warmup {k+1}/2: {s_iso} → {e_iso}")
            _ = post_simulation(sess, s_iso, e_iso)
            time.sleep(0.2)
        except Exception as e:
            print("warmup skip:", e)

def main():
    print(f"POST {URL}  (payload tz: {'UTC' if USE_UTC_PAYLOAD else 'CET'})")
    print(f"Start chunk={INITIAL_CHUNK_MIN}min, min chunk={MIN_CHUNK_MIN}min, timeout={REQUEST_TIMEOUT_S}s, n_profiles={N_PROFILES}")
    sess = session_with_retries()

    start_utc = datetime(YEAR, 1, 1, tzinfo=TZ_UTC)
    end_utc   = datetime(YEAR+1, 1, 1, tzinfo=TZ_UTC)

    warmup(sess, start_utc)  # <<< nieuw

    transitions_utc = detect_tz_transitions_utc(YEAR)

    chunks = []
    cur = start_utc
    chunk_min = INITIAL_CHUNK_MIN
    backoff_s = 1.0

    while cur < end_utc:
        nxt = min(cur + timedelta(minutes=chunk_min), end_utc)
        # knip op DST-transities
        for tr in transitions_utc:
            if cur < tr < nxt:
                nxt = tr
                break

        if USE_UTC_PAYLOAD:
            s_iso, e_iso = cur.isoformat(), nxt.isoformat()
        else:
            s_iso, e_iso = utc_to_local_iso(cur), utc_to_local_iso(nxt)

        try:
            print(f"- {s_iso} → {e_iso}  ({(nxt-cur).total_seconds()/60:.0f}m)")
            resp = post_simulation(sess, s_iso, e_iso)
            dfc = to_dataframe(resp)
            print(f"  ok: {dfc.shape[0]} timestamps × {max(dfc.shape[1]-1,0)} profielen")
            chunks.append(dfc)

            # succes → chunk vergroten tot max
            if chunk_min < MAX_CHUNK_HOURS*60:
                chunk_min = min(MAX_CHUNK_HOURS*60, max(chunk_min, chunk_min*2))
            backoff_s = 1.0
            cur = nxt
            if BETWEEN_CALLS_SLEEP_S:
                time.sleep(BETWEEN_CALLS_SLEEP_S)

        except Exception as e:
            msg = str(e)
            if is_transient(msg) and chunk_min > MIN_CHUNK_MIN:
                # halveer chunk en retry (zonder cur vooruit te zetten)
                chunk_min = max(MIN_CHUNK_MIN, chunk_min // 2)
                print(f"  transient/server issue → verklein chunk naar {chunk_min}min, backoff {backoff_s:.1f}s")
                time.sleep(backoff_s)
                backoff_s = min(30.0, backoff_s * 1.8)
                continue
            print("Fout:", e)
            sys.exit(1)

    if not any([not c.empty for c in chunks]):
        print("Geen data ontvangen.")
        sys.exit(1)

    wide = (pd.concat(chunks, axis=0, ignore_index=True)
            .drop_duplicates(subset=["timestamp_local"])
            .sort_values("timestamp_local"))
    wide.to_csv(OUT_WIDE, index=False)
    print(f"Geschreven (wide): {OUT_WIDE}  -> {wide.shape[0]} rijen, {wide.shape[1]-1} profielen")

    long = (wide.melt(id_vars=["timestamp_local"], var_name="cp_id", value_name="demand_kw")
                 .sort_values(["timestamp_local","cp_id"]))
    long.to_csv(OUT_LONG, index=False)
    print(f"Geschreven (long): {OUT_LONG}  -> {long.shape[0]} rijen")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fout:", e)
        sys.exit(1)
