# plot_epochs.py
# Plot avg business loss per epoch voor Q-learning en DQN
# Gebruik:
#   python plot_epochs.py --qlog qlearn_epoch_log.csv --dqnlog dqn_epoch_log.csv --out loss_by_epoch.png --smooth 1

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _load_epoch_series(path: Path, label: str):
    """
    Verwacht een CSV met ten minste kolommen:
      - 'epoch'
      - 'avg_loss'  (business loss: revenue - alpha * peak)
    Geeft (epochs_sorted, avg_loss_sorted, label) of None als bestand ontbreekt.
    """
    if not path.exists():
        print(f"⚠ {label}: bestand niet gevonden: {path}")
        return None

    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"{label}: kolom 'epoch' ontbreekt in {path}")
    if "avg_loss" not in df.columns:
        # fallback: probeer loss te herleiden uit daglog (niet ideaal)
        if {"revenue", "peak_kWh"}.issubset(df.columns):
            print(f"ℹ {label}: 'avg_loss' ontbreekt; herleid uit revenue/peak_kWh per rij.")
            df["avg_loss"] = -pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0) + \
                              2.0 * pd.to_numeric(df["peak_kWh"], errors="coerce").fillna(0.0)
        else:
            raise ValueError(f"{label}: kolom 'avg_loss' ontbreekt in {path}")

    # Sommige logs kunnen meerdere rijen per epoch hebben; neem gemiddelde per epoch
    g = (df.groupby("epoch", as_index=True)["avg_loss"]
           .mean()
           .sort_index())
    x = g.index.to_numpy()
    y = g.to_numpy(dtype=float)
    return (x, y, label)

def main():
    ap = argparse.ArgumentParser(description="Plot avg business loss per epoch voor Q-learning en DQN")
    ap.add_argument("--qlog",  type=str, default="qlearn_epoch_log.csv", help="Pad naar Q-learning epoch log")
    ap.add_argument("--dqnlog", type=str, default="dqn_epoch_log.csv",   help="Pad naar DQN epoch log")
    ap.add_argument("--out",   type=str, default="loss_by_epoch.png",    help="Uitvoerbestand (PNG)")
    ap.add_argument("--smooth", type=int, default=1, help="Rolling venster (epochs) voor smoothing; 1 = geen")
    ap.add_argument("--dpi",    type=int, default=140, help="DPI voor PNG")
    ap.add_argument("--show", action="store_true", help="Toon de plot interactief")
    args = ap.parse_args()

    series = []
    q = _load_epoch_series(Path(args.qlog),  "Q-learning")
    d = _load_epoch_series(Path(args.dqnlog), "DQN")
    if q: series.append(q)
    if d: series.append(d)

    if not series:
        raise SystemExit("Geen data om te plotten. Controleer paden naar logs.")

    plt.figure(figsize=(10, 5))

    for x, y, label in series:
        plt.plot(x, y, linewidth=1.5, label=label)
        if args.smooth and args.smooth > 1:
            # rolling over epochs: gebruik pandas voor gemakkelijke rolling
            ys = pd.Series(y, index=x).rolling(window=args.smooth, min_periods=1).mean()
            plt.plot(ys.index.to_numpy(), ys.values, linewidth=2.5, linestyle="--",
                     label=f"{label} (rolling {args.smooth})")

    plt.xlabel("Epoch")
    plt.ylabel("Average loss  (−revenue + α·peak)")
    plt.title("Averag loss per epoch: Q-learning vs DQN")
    plt.legend(loc="best")
    plt.tight_layout()

    out = Path(args.out)
    plt.savefig(out, dpi=args.dpi)
    print(f"✅ Plot opgeslagen: {out.resolve()}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
