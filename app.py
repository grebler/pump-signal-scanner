import time, math, requests
import pandas as pd
import numpy as np
from dateutil import tz

# -------- CONFIG --------
PAIRS_PER_SCAN      = 20        # newest pairs to check each loop
INTERVAL_SECONDS     = 30       # scan cadence
CANDLE_INTERVAL      = "1m"     # 1m candles (DexScreener)
LOOKBACK             = 120      # candles to compute indicators
MIN_ABS_VOLUME_USD   = 3000     # min avg(20) USD volume
MIN_LIQ_USD          = 30000    # min liquidity
ROC_WINDOW           = 5        # bars for market-cap ROC
VOL_MULT_CONFIRM     = 1.5      # vol must be >1.5x 20-bar avg
MIN_SIGNALS_REQUIRED = 2        # fire alert if >=2 rules pass

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID= os.getenv("TELEGRAM_CHAT_ID")

# -------- HELPERS --------
def send_tg(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; skipping message.")
        return
    try:
        requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            params={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print("Telegram error:", e)

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def bollinger(series, n=20, k=2):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    return ma, ma + k*sd, ma - k*sd

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100/(1+rs))

def obv(close, volume):
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).fillna(0).cumsum()

# -------- DATA (DexScreener) --------
def get_new_solana_pairs():
    # New pairs endpoint (frequently updated)
    url = "https://api.dexscreener.com/latest/dex/pairs/solana"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json().get("pairs", [])
    # Sort newest first
    data.sort(key=lambda d: d.get("pairCreatedAt", 0), reverse=True)
    return data

def get_candles(pair_address, interval=CANDLE_INTERVAL, lookback=LOOKBACK):
    # Candle endpoint
    url = f"https://api.dexscreener.com/latest/dex/candles/solana/{interval}?pairAddress={pair_address}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return pd.DataFrame()
    candles = r.json().get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)[-lookback:]
    # normalize columns
    rename = {"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"}
    df = df.rename(columns=rename)
    # volume is base-token amount; try USD volume & liquidity via pair snapshot too
    return df

def enrich_with_snapshot(df, pair):
    # Add USD metrics if present from pair snapshot (approx)
    liq = pair.get("liquidity", {})
    quote = pair.get("priceUsd")
    # best effort: set liquidityUsd column with totalUSD if available
    liq_usd = (liq.get("usd", None) or liq.get("base", None) or 0)
    df["liquidityUsd"] = liq_usd
    # approximate volumeUsd per bar if priceUsd provided
    try:
        px = float(quote) if quote else np.nan
    except:
        px = np.nan
    df["volumeUsd"] = df["volume"] * px
    # rough market cap from FDV if available
    try:
        mcap = float(pair.get("fdv", None) or pair.get("marketCap", None) or np.nan)
    except:
        mcap = np.nan
    df["mcap"] = mcap
    return df

# -------- RULES --------
def rule_ema_cross(df):
    c = df["close"]
    v = df["volumeUsd"].fillna(0)
    e9, e21 = ema(c, 9), ema(c, 21)
    cross = (e9.iloc[-2] < e21.iloc[-2]) and (e9.iloc[-1] > e21.iloc[-1])
    vavg = v.rolling(20).mean().iloc[-1] or 1
    vol_ok = v.iloc[-1] > VOL_MULT_CONFIRM * vavg
    return bool(cross and vol_ok), {"vol_mult": float(v.iloc[-1]/vavg)}

def rule_boll_breakout(df):
    c = df["close"]
    ma, up, dn = bollinger(c, 20, 2)
    bw = (up - dn) / ma
    # squeeze if current bandwidth in the lowest 20% of last 30 bars
    win = bw.rolling(30)
    pct20 = np.nanpercentile(win.apply(lambda x: x.iloc[-1], raw=False).dropna(), 20) if win.count().iloc[-1] >= 30 else np.nan
    squeeze = (not math.isnan(pct20)) and (bw.iloc[-1] <= pct20)
    breakout = c.iloc[-1] > (up.iloc[-1] if not math.isnan(up.iloc[-1]) else np.inf)
    return bool(squeeze and breakout), {}

def rule_rsi_reclaim(df, level=50):
    r = rsi(df["close"]).iloc[-1]
    return bool(r > level), {"rsi": float(r)}

def rule_obv_leads(df):
    c = df["close"]; v = df["volume"]
    o = obv(c, v)
    price_hh = c.iloc[-1] > c.rolling(50).max().iloc[-2]
    obv_hh   = o.iloc[-1] > o.rolling(50).max().iloc[-2]
    return bool(obv_hh and not price_hh), {}

def rule_mcap_roc(df, thr=5):
    if "mcap" not in df or df["mcap"].isna().all():
        return False, {}
    roc = df["mcap"].pct_change(ROC_WINDOW).iloc[-1] * 100
    return bool(roc > thr), {"roc%": float(roc)}

def guards(df):
    vcol = "volumeUsd" if "volumeUsd" in df else "volume"
    vavg = df[vcol].rolling(20).mean().iloc[-1]
    vol_ok = (vavg is not None) and (vavg >= MIN_ABS_VOLUME_USD)
    liq_ok = ("liquidityUsd" in df) and (df["liquidityUsd"].iloc[-1] >= MIN_LIQ_USD)
    return bool(vol_ok and liq_ok)

# -------- MAIN LOOP --------
def scan_once():
    pairs = get_new_solana_pairs()
    if not pairs:
        return
    for pair in pairs[:PAIRS_PER_SCAN]:
        addr = pair.get("pairAddress") or pair.get("address")
        sym  = pair.get("baseToken", {}).get("symbol") or pair.get("baseToken", {}).get("name") or "?"
        if not addr:
            continue
        df = get_candles(addr)
        if df.empty or len(df) < 60:
            continue
        df = enrich_with_snapshot(df, pair)
        if not guards(df):
            continue

        hits = []
        for f in (rule_ema_cross, rule_boll_breakout, rule_rsi_reclaim, rule_obv_leads, rule_mcap_roc):
            try:
                ok, meta = f(df)
            except Exception:
                ok, meta = False, {}
            if ok: hits.append(f.__name__)

        if len(hits) >= MIN_SIGNALS_REQUIRED:
            last = df.iloc[-1]
            msg = (
                f"<b>{sym}</b>  (signals: {', '.join(hits)})\n"
                f"Price: {last['close']:.6f} | Vol20avg: {df['volumeUsd'].rolling(20).mean().iloc[-1]:.0f} USD\n"
                f"Liq: {int(df['liquidityUsd'].iloc[-1])} USD  |  Mcap ROC({ROC_WINDOW}): "
                f"{'n/a' if df['mcap'].isna().all() else round(df['mcap'].pct_change(ROC_WINDOW).iloc[-1]*100,2)}%\n"
                f"Playbook: TP +20% | TSL 0.87x"
            )
            print(msg.replace("<b>","").replace("</b>",""))
            send_tg(msg)

def main():
    print("Scanner starting…")
    while True:
        try:
            scan_once()
        except Exception as e:
            print("Loop error:", e)
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    import os
    main()
