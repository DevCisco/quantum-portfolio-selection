"""Download monthly adjusted closing prices for the stocks in universe.csv.

Run on your own computer (needs internet):
    pip install yfinance pandas
    python download_prices.py

Version 2: daily data resampled to month-end; empty tickers retried.

Output: prices_monthly.csv (one column per ticker, month-end adjusted close in EUR)
        download_log.txt  (what was downloaded, when, and any gaps)

Yahoo Finance data are for personal/research use: do NOT commit prices_monthly.csv
to a public repository; commit this script and universe.csv instead.
"""
import datetime as dt
import pandas as pd
import yfinance as yf

START, END = "2014-01-01", "2026-09-01"   # 12 months of history before 2015 for the first estimation window

universe = pd.read_csv("universe.csv")
tickers = universe["ticker"].tolist()

FALLBACK = {"ALV.DE": ["ALV.DE", "ALV.F", "ALIZF"]}   # Allianz came back empty in version 1


def monthly_close(symbols):
    raw = yf.download(symbols, start=START, end=END, interval="1d", auto_adjust=True, progress=True, group_by="column")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.resample("ME").last()


prices = monthly_close(tickers)[tickers]
notes = []
for t in tickers:
    if prices[t].notna().sum() > 0:
        continue
    for alt in FALLBACK.get(t, [t]):          # retry alone, then alternative listings
        s = monthly_close([alt])[alt].reindex(prices.index)
        if s.notna().sum() > 0:
            prices[t] = s
            notes.append(f"{t}: empty in bulk download, filled from '{alt}'"
                         + (" (different listing/currency: check before use)" if alt != t else ""))
            break
    else:
        notes.append(f"{t}: no data from any listing -> drop it from universe.csv or replace it")
prices.to_csv("prices_monthly.csv", float_format="%.6f")

with open("download_log.txt", "w") as fh:
    fh.write(f"downloaded: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    fh.write(f"yfinance version: {yf.__version__}\n")
    fh.write(f"period: {prices.index.min().date()} .. {prices.index.max().date()}  ({len(prices)} months)\n")
    for n_ in notes:
        fh.write(f"NOTE {n_}\n")
    for t in tickers:
        s = prices[t]
        fh.write(f"{t:10s} first={s.first_valid_index().date() if s.notna().any() else 'NONE'}  missing={int(s.isna().sum())}\n")
print(open("download_log.txt").read())
