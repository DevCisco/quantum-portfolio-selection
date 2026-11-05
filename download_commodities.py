"""Download monthly commodity prices (continuous front-month futures on Yahoo Finance),
convert them from USD to EUR, and flag anomalous months.

Run on your own computer (needs internet):
    pip install yfinance pandas
    python download_commodities.py

Input : commodities.csv  (set include=1/0 to choose the series; uranium is off by default)
Version 2: daily data resampled to month-end (version 1 lost 22 months).

Output: commodities_monthly_eur.csv   month-end prices in EUR, one column per ticker
        commodities_monthly_usd.csv   the same in USD (as downloaded)
        commodities_log.txt           download date, coverage, and flagged months

Notes
- Continuous futures jump at each contract roll; the jump is not a return an investor earned.
  Monthly sampling reduces but does not remove this effect (it is strongest for energy).
- WTI crude settled below zero in April 2020: non-positive prices are flagged and set to NaN.
- Yahoo Finance data are for personal/research use: do NOT commit the output CSVs to a
  public repository; commit this script and commodities.csv instead.
"""
import datetime as dt
import pandas as pd
import yfinance as yf

START, END = "2014-01-01", "2026-09-01"
JUMP_THRESHOLD = 0.5       # flag |monthly return| above 50 %

table = pd.read_csv("commodities.csv")
table = table[table["include"] == 1]
tickers = table["ticker"].tolist()


def monthly_close(symbols):
    """Daily closes resampled to the last trading day of each month. Yahoo's own monthly bars for
    futures are sometimes dated mid-month, which made whole months disappear in version 1."""
    raw = yf.download(symbols, start=START, end=END, interval="1d", auto_adjust=False, progress=True, group_by="column")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.resample("ME").last()


usd = monthly_close(tickers)[tickers]
fx = monthly_close(["EURUSD=X"])["EURUSD=X"].reindex(usd.index)     # USD per 1 EUR

flags = []
for t in tickers:
    bad = usd[t] <= 0
    for d in usd.index[bad]:
        flags.append(f"{t:8s} {d.date()}  non-positive price {usd.at[d, t]:.4f} -> set to NaN")
    usd.loc[bad, t] = float("nan")
    r = usd[t].pct_change(fill_method=None)
    for d in r.index[r.abs() > JUMP_THRESHOLD]:
        flags.append(f"{t:8s} {d.date()}  monthly change {r.at[d]:+.1%} (possible roll jump or extreme move)")

eur = usd.div(fx, axis=0)
usd.to_csv("commodities_monthly_usd.csv", float_format="%.6f")
eur.to_csv("commodities_monthly_eur.csv", float_format="%.6f")

with open("commodities_log.txt", "w") as fh:
    fh.write(f"downloaded: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    fh.write(f"yfinance version: {yf.__version__}\n")
    fh.write(f"period: {usd.index.min().date()} .. {usd.index.max().date()}  ({len(usd)} months)\n")
    full = pd.date_range(usd.index.min(), usd.index.max(), freq="ME")
    fh.write(f"EURUSD missing months: {int(fx.isna().sum())}\n")
    fh.write(f"calendar months missing from the index: {len(full.difference(usd.index))}\n\ncoverage:\n")
    for t in tickers:
        s = eur[t]
        first = s.first_valid_index()
        fh.write(f"  {t:8s} first={first.date() if first is not None else 'NONE'}  missing={int(s.isna().sum())}\n")
    fh.write("\nflags:\n" + ("\n".join("  " + f for f in flags) if flags else "  none") + "\n")
print(open("commodities_log.txt").read())
