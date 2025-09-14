rgparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot Q-learning loss vs days")
    parser.add_argument("--log", type=str, default="qlearn_training_log.csv",
                        help="Pad naar training log CSV (default: qlearn_training_log.csv)")
    parser.add_argument("--out", type=str, default="loss_over_days.png",
                        help="Uitvoerbestand voor de figuur (PNG)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling window (in dagen) voor smoothing; 1 = geen smoothing")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="ALPHA (alleen gebruikt als 'loss' kolom ontbreekt en we loss herberekenen)")
    parser.add_argument("--dpi", type=int, default=140, help="DPI voor de PNG")
    parser.add_argument("--show", action="store_true", help="Toon de plot interactief")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Bestand niet gevonden: {log_path}")

    # Inlezen
    df = pd.read_csv(log_path)

    # Verwachte kolommen: epoch, date, revenue, peak_kWh, loss, epsilon
    # Datum parsen en sorteren op epoch + date indien aanwezig
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass

    sort_cols = []
    if "epoch" in df.columns:
        sort_cols.append("epoch")
    if "date" in df.columns:
        sort_cols.append("date")

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Dagindex (1..N) over de volledige trainingsrun
    df["day_num"] = np.arange(1, len(df) + 1)

    # Loss aanwezig? Zo niet, herbereken met -revenue + alpha * peak_kWh
    if "loss" not in df.columns or df["loss"].isna().all():
        if not {"revenue", "peak_kWh"}.issubset(df.columns):
            raise SystemExit("Geen 'loss' kolom en onvoldoende kolommen om loss te herberekenen "
                             "(vereist: revenue en peak_kWh).")
        df["loss"] = -pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0) + \
                      args.alpha * pd.to_numeric(df["peak_kWh"], errors="coerce").fillna(0.0)

    # Smoothing (rolling mean)
    if args.smooth > 1:
        df["loss_smooth"] = df["loss"].rolling(args.smooth, min_periods=1).mean()
    else:
        df["loss_smooth"] = df["loss"]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["day_num"], df["loss"], linewidth=1, label="Loss (dagelijks)")
    if args.smooth > 1:
        plt.plot(df["day_num"], df["loss_smooth"], linewidth=2, label=f"Loss (rolling {args.smooth}d)")

    # Verticale lijnen bij epoch-wissels (optioneel, indien epoch-kolom bestaat)
    if "epoch" in df.columns:
        epochs = df["epoch"].values
        # Vind indices waar epoch verandert
        change_idx = np.flatnonzero(np.diff(epochs)) + 1
        for ci in change_idx:
            x = df.loc[ci, "day_num"]
            plt.axvline(x=x, linestyle="--", linewidth=0.8)

    plt.xlabel("Dagen (sequentieel)")
    plt.ylabel("Loss")
    plt.title("Q-learning training loss per dag")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = Path(args.out)
    plt.savefig(out_path, dpi=args.dpi)
    print(f"✅ Plot opgeslagen: {out_path.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()