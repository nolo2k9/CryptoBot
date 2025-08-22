#!/usr/bin/env python3
"""
tradebot_pro.py — Binance spot trading bot with:
- Paper/live/backtest modes
- Env-based API keys (or CLI), optional testnet in paper mode
- Exchange rule enforcement (LOT_SIZE, NOTIONAL, PRICE_FILTER)
- Dynamic position sizing via ATR
- RSI, ADX, MACD, Bollinger Bands, VWAP, Stochastic RSI, Volume Oscillator indicators
- Optional lightweight ML blend (LogisticRegression)
- Idempotent orders (clientOrderId)
- Exponential backoff w/ jitter on API calls
- Structured logging + optional CSV trade log
- Real stop-loss / take-profit exits
- Per-symbol PnL tracking (realized/unrealized) + portfolio daily kill-switch
- Auto-select symbols: top gainer / top loser / both
- Backtest with same SL/TP rules and summary metrics
- WebSocket for real-time data
- Portfolio-level VaR risk management
- Prometheus metrics for real-time monitoring
- Email alerts for critical events
- NEW: Fallback to BTCUSDT if no valid symbols
- NEW: WebSocket stream validation to prevent KeyError
- NEW: Fix DataFeed client attribute error

Requires: python-binance, pandas, numpy, python-dotenv, websocket-client, prometheus-client, smtplib, (optional) scikit-learn
Example:
  python tradebot_pro.py --symbols BTCUSDT ETHUSDT --interval 5m --mode paper
  python tradebot_pro.py --auto-select both --interval 5m --mode paper --min-volume 5e7
  python tradebot_pro.py --symbols BTCUSDT --interval 5m --mode backtest --days 60

Env vars:
  BINANCE_KEY, BINANCE_SECRET, EMAIL_USER, EMAIL_PASS, EMAIL_RECIPIENT
  MODE=paper|live|backtest
  LOG_FILE=trader.log
  TRADE_LOG_CSV=trades.csv
  RISK_PER_TRADE=0.01
  DAILY_LOSS_LIMIT=-0.05
  PORTFOLIO_VAR_LIMIT=-0.05
  MAX_TRADE_HOURS=24
  USE_TESTNET=true
  FEE_RATE=0.001
  TP_MULT=3.0
  SL_MULT=2.0
  FORCE_DEFAULT=true
"""

from dotenv import load_dotenv
import os
import sys
import time
import ta
import json
import uuid
import random
import logging
import smtplib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd
import websocket
import threading

from binance.client import Client
from binance.exceptions import BinanceAPIException
from prometheus_client import Gauge, start_http_server

# Optional ML
try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# NEW: Rate-limiting for email alerts
last_alert_times = {}  # Track last alert time per event type

# Load env
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

# -------- Logging and Metrics --------
def setup_logging(log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,  # change DEBUG to INFO or WARNING
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers
    )

# NEW: Prometheus metrics
equity_gauge = Gauge("bot_equity_usdt", "Current portfolio equity")
pnl_gauge = Gauge("bot_realized_pnl_usdt", "Realized PnL", ["symbol"])
drawdown_gauge = Gauge("bot_max_drawdown_pct", "Maximum drawdown percentage")

# NEW: Email alerts with rate-limiting
def send_alert(subject: str, body: dict):
    try:
        email_user = os.getenv("EMAIL_USER")
        email_pass = os.getenv("EMAIL_PASS")
        recipient = os.getenv("EMAIL_RECIPIENT")
        if not all([email_user, email_pass, recipient]):
            logging.warning("Email config missing; skipping alert")
            return
        # Rate-limit emails
        event_type = body.get("Event", subject)
        now = datetime.now(timezone.utc)
        last_alert = last_alert_times.get(event_type, now - timedelta(minutes=1))
        if (now - last_alert).total_seconds() < 60:
            logging.info(f"Skipping email alert for {event_type} (rate-limited)")
            return
        last_alert_times[event_type] = now
        # Format body as a readable string
        formatted_body = "\n".join(f"{key}: {value}" for key, value in body.items())
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_user, email_pass)
            msg = f"Subject: {subject}\n\n{formatted_body}"
            server.sendmail(email_user, recipient, msg)
        logging.info(f"Sent email alert: {subject}")
    except Exception as e:
        logging.warning(f"Failed to send email alert: {e}")

def get_min_order(symbol_info):
    """
    Returns minimum notional required for an order.
    """
    for f in symbol_info['filters']:
        if f['filterType'] == 'MIN_NOTIONAL':
            return float(f['minNotional'])
    return 10.0  # fallback default

def format_quantity(qty: float, step_size: float) -> str:
    # Compute decimal places from step size
    decimal_places = max(0, -Decimal(str(step_size)).as_tuple().exponent)
    fmt_str = f"{{:.{decimal_places}f}}"
    return fmt_str.format(qty)


def getenv_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

# -------- Binance Client Wrapper --------
class SafeBinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self._symbol_info_cache: Dict[str, dict] = {}
        self.testnet = testnet

    def _backoff(self, attempt: int):
        time.sleep(min(60, (2 ** attempt) + random.random()))

    def call(self, fn, *args, **kwargs):
        for attempt in range(6):
            try:
                return fn(*args, **kwargs)
            except BinanceAPIException as e:
                logging.warning(f"BinanceAPIException (attempt {attempt + 1}): {e}")
                if attempt == 5:
                    raise
                self._backoff(attempt)
            except Exception as e:
                logging.warning(f"Unexpected exception (attempt {attempt + 1}): {e}")
                if attempt == 5:
                    raise
                self._backoff(attempt)

    def get_account(self):
        return self.call(self.client.get_account)

    def get_symbol_info(self, symbol: str):
        if symbol not in self._symbol_info_cache:
            self._symbol_info_cache[symbol] = self.call(self.client.get_symbol_info, symbol)
        return self._symbol_info_cache[symbol]

    def get_klines(self, symbol: str, interval: str, startTime=None, endTime=None, limit=1000):
        return self.call(self.client.get_klines, symbol=symbol, interval=interval,
                         startTime=startTime, endTime=endTime, limit=limit)

    def get_ticker(self):
        return self.call(self.client.get_ticker)

    def create_order(self, **kwargs):
        return self.call(self.client.create_order, **kwargs)

    def create_test_order(self, **kwargs):
        return self.call(self.client.create_test_order, **kwargs)

# -------- Exchange filters helpers --------
def decimal_truncate_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    d = Decimal(str(step))
    places = max(0, -d.as_tuple().exponent)
    q = Decimal(10) ** (-places)
    return float((Decimal(x).quantize(q, rounding=ROUND_DOWN)))

def enforce_exchange_filters(client: SafeBinanceClient, symbol: str,
                             qty: float, price: Optional[float]) -> Tuple[float, Optional[float], float]:
    info = client.get_symbol_info(symbol)
    if not info:
        raise RuntimeError(f"Symbol info unavailable for {symbol}")

    filters = {f["filterType"]: f for f in info.get("filters", [])}
    lot = filters.get("LOT_SIZE", {})
    step_size = float(lot.get("stepSize", "0.00000001"))
    min_qty = float(lot.get("minQty", "0.0"))
    max_qty = float(lot.get("maxQty", "1e9"))
    adj_qty = max(min(decimal_truncate_to_step(qty, step_size), max_qty), min_qty)

    adj_price = price
    if price is not None and "PRICE_FILTER" in filters:
        pf = filters["PRICE_FILTER"]
        tick_size = float(pf.get("tickSize", "0.00000001"))
        adj_price = decimal_truncate_to_step(price, tick_size)

    notional = 0.0
    if "NOTIONAL" in filters:
        notional = float(filters["NOTIONAL"].get("minNotional", "0.0"))

    return adj_qty, adj_price, notional

# -------- Indicators --------
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tr = np.maximum(out['High'] - out['Low'],
                    np.maximum(np.abs(out['High'] - out['Close'].shift()),
                               np.abs(out['Low'] - out['Close'].shift())))
    out['ATR'] = tr.rolling(window=14, min_periods=1).mean()

    delta = out['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    out['RSI'] = 100 - (100 / (1 + rs))

    up = out['High'].diff()
    down = -out['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr14)
    dx = (100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    out['ADX'] = dx.rolling(14).mean()

    exp1 = out['Close'].ewm(span=12, adjust=False).mean()
    exp2 = out['Close'].ewm(span=26, adjust=False).mean()
    out['MACD'] = exp1 - exp2
    out['MACD_SIGNAL'] = out['MACD'].ewm(span=9, adjust=False).mean()

    ma20 = out['Close'].rolling(window=20).mean()
    std20 = out['Close'].rolling(window=20).std()
    out['BB_UPPER'] = ma20 + 2 * std20
    out['BB_LOWER'] = ma20 - 2 * std20

    # NEW: Add VWAP
    out['VWAP'] = (out['Close'] * out['Volume']).cumsum() / out['Volume'].cumsum()
    # NEW: Stochastic RSI
    k = 14
    smooth_k = 3
    smooth_d = 3
    rsi = out['RSI']
    rsi_low = rsi.rolling(k).min()
    rsi_high = rsi.rolling(k).max()
    out['Stoch_RSI'] = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-9)
    out['Stoch_RSI_K'] = out['Stoch_RSI'].rolling(smooth_k).mean()
    out['Stoch_RSI_D'] = out['Stoch_RSI_K'].rolling(smooth_d).mean()
    # NEW: Volume Oscillator
    short_vol = out['Volume'].ewm(span=5, adjust=False).mean()
    long_vol = out['Volume'].ewm(span=10, adjust=False).mean()
    out['Vol_Osc'] = 100 * (short_vol - long_vol) / long_vol
    return out

def simple_signal(row) -> int:
    long = (row['Close'] > row.get('BB_LOWER', 0)) and (row['RSI'] > 55) and (row['MACD'] > row['MACD_SIGNAL'])
    short = (row['Close'] < row.get('BB_UPPER', 0)) and (row['RSI'] < 45) and (row['MACD'] < row['MACD_SIGNAL'])
    if long and not short:
        return 1
    if short and not long:
        return -1
    return 0

# -------- Optional ML --------
class RollingML:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000) if _HAS_SK else None
        self.trained = False

    def fit(self, df: pd.DataFrame):
        if self.model is None:
            return
        feats = df[['RSI', 'MACD', 'MACD_SIGNAL', 'ATR']].fillna(0.0).values
        future_ret = df['Close'].pct_change().shift(-1).fillna(0.0)
        y = (future_ret > 0).astype(int).values
        self.model.fit(feats, y)
        self.trained = True

    def predict_signal(self, row) -> Optional[int]:
        if not self.trained or self.model is None:
            return None
        x = np.array([[row['RSI'], row['MACD'], row['MACD_SIGNAL'], row['ATR']]])
        proba = self.model.predict_proba(x)[0, 1]
        if proba > 0.55:
            return 1
        if proba < 0.45:
            return -1
        return 0

# -------- Data helpers --------
def fetch_klines_series(client: SafeBinanceClient, symbol: str, interval: str, start: datetime,
                       end: datetime) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    out = []
    last = start_ms
    while True:
        chunk = client.get_klines(symbol, interval, startTime=last, endTime=end_ms, limit=1000)
        if not chunk:
            break
        out.extend(chunk)
        last_open = chunk[-1][0]
        if len(chunk) < 1000 or last_open >= end_ms:
            break
        last = last_open + 1
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
    ])
    df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.set_index("Date", inplace=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df

# NEW: WebSocket Data Feed
class DataFeed:
    def __init__(self, symbols: List[str], interval: str, client: Client):
        self.client = client
        self.symbols = [sym.upper() for sym in symbols]
        self.interval = interval
        self.data: Dict[str, pd.DataFrame] = {}
        self.ws = None
        self._stop = False

        for sym in self.symbols:
            try:
                info = client.get_symbol_info(sym)
                df = fetch_klines_series(client, sym, interval, datetime.utcnow() - timedelta(days=1), datetime.utcnow())
                self.data[sym] = df
                print(f"Initialized WebSocket data for {sym} ({'with history' if not df.empty else 'empty'})")
            except BinanceAPIException as e:
                print(f"Cannot initialize WebSocket for {sym}: {e}")

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            if 'k' in msg:
                k = msg['k']
                sym = k['s']
                if sym in self.data:
                    df_new = pd.DataFrame({
                        'Date': [pd.to_datetime(k['t'], unit='ms')],
                        'Open': [float(k['o'])],
                        'High': [float(k['h'])],
                        'Low': [float(k['l'])],
                        'Close': [float(k['c'])],
                        'Volume': [float(k['v'])]
                    }).set_index('Date')
                    self.data[sym] = pd.concat([self.data[sym], df_new]).tail(200)
                    self.data[sym] = indicators(self.data[sym])
                    print(f"Received WebSocket data for {sym}")
        except Exception as ex:
            print(f"WebSocket message processing error: {ex}")

    def on_open(self, ws):
        print("WebSocket connected")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")

    def run_forever_with_reconnect(self, max_retries=10):
        retry = 0
        while not self._stop and retry < max_retries:
            streams = '/'.join(f"{sym.lower()}@kline_{self.interval}" for sym in self.symbols)
            # Use mainnet websocket for all modes to ensure stable price data feed
            base_url = f"wss://stream.binance.com:9443/stream?streams={streams}"
            print(f"Connecting to {base_url} (Attempt {retry + 1})")

            self.ws = websocket.WebSocketApp(
                base_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            try:
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"WebSocket run error: {e}")
            backoff = min(60, 2 ** retry) + random.uniform(0, 1)
            print(f"Reconnect attempt {retry + 1} failed, retrying in {backoff:.2f} seconds...")
            time.sleep(backoff)
            retry += 1

        print("Max reconnect attempts reached, stopping websocket.")

    def start(self):
        import threading
        thread = threading.Thread(target=self.run_forever_with_reconnect, daemon=True)
        thread.start()


# NEW: Portfolio VaR
def portfolio_var(positions: Dict[str, dict], prices: Dict[str, float], df: Dict[str, pd.DataFrame]) -> float:
    weights = []
    returns = []
    for sym, pos in positions.items():
        if pos["qty"] > 0:
            weight = pos["qty"] * prices.get(sym, pos["entry_price"]) / sum(
                pos["qty"] * prices.get(s, pos["entry_price"]) for s, pos in positions.items() if pos["qty"] > 0)
            weights.append(weight)
            rets = df[sym]['Close'].pct_change().dropna() if sym in df and not df[sym].empty else pd.Series()
            returns.append(rets[-252:])  # ~1 year of 1h candles
    if not weights or not returns:
        return 0.0
    cov_matrix = np.cov(returns)
    portfolio_var = np.sqrt(np.dot(weights, np.dot(cov_matrix * 252, weights)))  # Annualized VaR
    return portfolio_var

# -------- Sizing / PnL helpers --------
def position_size_from_atr(balance_usdt: float, price: float, atr: float, risk_per_trade: float) -> float:
    if atr <= 0 or price <= 0:
        return 0.0
    risk_dollars = balance_usdt * risk_per_trade
    size_dollars = max(0.0, risk_dollars / max(atr, 1e-9))
    qty = size_dollars / price
    return max(qty, 0.0)

def get_free_usdt(client: SafeBinanceClient) -> float:
    acct = client.get_account()
    bals = acct.get('balances', [])
    for b in bals:
        if b.get('asset') in ('USDT', 'BUSD', 'USDC'):
            return float(b.get('free', 0.0))
    return 0.0

def order_notional(side: str, price: float, qty: float) -> float:
    return abs(price * qty)

def trade_pnl(side: str, entry: float, exit: float, qty: float) -> float:
    if side == "BUY":
        return (exit - entry) * qty
    else:
        return (entry - exit) * qty

# -------- Orders --------
def place_market_order(client: SafeBinanceClient, symbol: str, side: str, qty: float, mode: str) -> dict:
    cid = f"tbpro-{uuid.uuid4().hex[:20]}"
    order_kwargs = dict(symbol=symbol, side=side, type="MARKET", quantity=qty, newClientOrderId=cid)
    if mode == "live":
        return client.create_order(**order_kwargs)
    else:
        client.create_test_order(**order_kwargs)
        return {"status": "TEST_ORDER_ACCEPTED", "clientOrderId": cid, "symbol": symbol, "side": side, "type": "MARKET", "executedQty": qty}

# -------- Symbol auto-select --------
def auto_select_symbols(client: SafeBinanceClient, mode: str, min_volume: float):
    tickers = client.get_ticker()
    usdt_pairs = [t for t in tickers if t.get('symbol', '').endswith("USDT")]
    usdt_pairs = [t for t in usdt_pairs if float(t.get('quoteVolume', 0.0)) >= float(min_volume)]
    if not usdt_pairs:
        logging.error("No USDT pairs meet minimum volume requirement")
        return []
    # Validate symbols for testnet compatibility
    valid_pairs = []
    for t in usdt_pairs:
        try:
            client.get_symbol_info(t['symbol'])
            client.get_klines(t['symbol'], Client.KLINE_INTERVAL_1MINUTE, limit=1)
            valid_pairs.append(t)
        except BinanceAPIException as e:
            logging.warning(f"Symbol {t['symbol']} not supported: {e}")
    if not valid_pairs:
        logging.error("No valid USDT pairs after validation")
        return []
    valid_pairs.sort(key=lambda x: float(x.get('priceChangePercent', 0.0)), reverse=True)
    if mode == "top_gainer":
        return [valid_pairs[0]['symbol']]
    elif mode == "top_loser":
        return [valid_pairs[-1]['symbol']]
    elif mode == "both":
        return [valid_pairs[0]['symbol'], valid_pairs[-1]['symbol']]
    else:
        return []

# -------- CSV trade logger --------
def append_trade_csv(csv_path: str, record: dict):
    try:
        df = pd.DataFrame([record])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    except Exception as e:
        logging.warning(f"Failed to write trade CSV: {e}")
def fetch_candle_data(client: SafeBinanceClient, symbol: str, interval: str = '1m', limit: int = 500):
    """
    Fetches recent candlestick data from Binance and returns a DataFrame.
    """
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    return df
# -------- Live/Paper loop --------
def run_live_or_paper(symbols: List[str], interval: str, mode: str,
                      risk_per_trade: float, daily_loss_limit: float, var_limit: float, max_hours: int,
                      use_testnet: bool, log_file: Optional[str] = None,
                      use_ml: bool = True, auto_select: Optional[str] = None,
                      min_volume: float = 5e7, fee_rate: float = 0.001,
                      tp_mult: float = 3.0, sl_mult: float = 2.0,
                      trade_log_csv: Optional[str] = None, force_default: bool = True):
    setup_logging(log_file)
    api_key = os.getenv("BINANCE_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET", "")
    if not api_key or not api_secret:
        logging.error("Missing BINANCE_KEY / BINANCE_SECRET")
        sys.exit(2)

    client = SafeBinanceClient(api_key, api_secret, testnet=(use_testnet and mode != "live"))

    # Auto-select symbols if requested
    if auto_select:
        symbols = auto_select_symbols(client, auto_select, min_volume)
        logging.info(f"Auto-selected symbols: {symbols}")

    # Validate symbols once before proceeding
    original_symbols = symbols.copy()
    valid_symbols = []
    for sym in symbols:
        try:
            client.get_symbol_info(sym)
            client.get_klines(sym, interval, limit=1)
            valid_symbols.append(sym)
        except BinanceAPIException as e:
            logging.warning(f"Symbol {sym} is invalid or not supported: {e}")
    symbols = valid_symbols

    # Fallback to BTCUSDT if no valid symbol found
    if not symbols and force_default:
        logging.warning(f"No valid symbols from {original_symbols}; falling back to BTCUSDT")
        try:
            client.get_symbol_info("BTCUSDT")
            client.get_klines("BTCUSDT", interval, limit=1)
            symbols = ["BTCUSDT"]
        except BinanceAPIException as e:
            logging.error(f"Default symbol BTCUSDT also invalid: {e}")
            sys.exit(1)

    if not symbols:
        logging.error(f"No valid symbols to trade: {original_symbols}")
        sys.exit(1)

    # Initialize DataFeed once with validated symbols
    data_feed = DataFeed(symbols, interval, client)

    # Per-symbol state initialization
    state = {
        sym: {
            "position": 0,
            "qty": 0.0,
            "entry_price": None,
            "entry_time": None,
            "stop": None,
            "take": None,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0
        } for sym in symbols
    }

    ml = RollingML() if use_ml else None

    # Daily kill-switch baseline setup
    day_start = datetime.now(timezone.utc)
    starting_day_balance = max(1.0, get_free_usdt(client))
    day_realized_pnl = 0.0
    peak_equity = starting_day_balance

    # Email alert on bot startup
    send_alert("TradeBot Pro Started", {
        "Event": "Bot Start",
        "Mode": mode,
        "Symbols": ", ".join(symbols),
        "Interval": interval,
        "Risk Per Trade": f"{risk_per_trade:.2%}",
        "Daily Loss Limit": f"{daily_loss_limit:.2%}",
        "Portfolio VaR Limit": f"{var_limit:.2%}",
        "Max Trade Hours": max_hours,
        "Use Testnet": use_testnet,
        "Use ML": use_ml,
        "Timestamp": datetime.now(timezone.utc).isoformat()
    })

    # Start Prometheus server
    start_http_server(8000)

    # Start WebSocket feed with reconnect logic (runs in background thread)
    data_feed.start()

    # Wait for WebSocket data to populate
    max_wait = 60
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if all(sym in data_feed.data and not data_feed.data[sym].empty for sym in symbols):
            break
        logging.info("Waiting for WebSocket data to populate...")
        time.sleep(5)

    if any(sym not in data_feed.data or data_feed.data[sym].empty for sym in symbols):
        unavailable = [sym for sym in symbols if sym not in data_feed.data or data_feed.data[sym].empty]
        logging.warning(f"WebSocket data missing for: {unavailable}")
        if force_default:
            logging.warning("Attempting fallback to BTCUSDT due to missing WebSocket data")
            try:
                client.get_symbol_info("BTCUSDT")
                client.get_klines("BTCUSDT", interval, limit=1)
                if "BTCUSDT" not in symbols:
                    symbols = ["BTCUSDT"]
                    data_feed = DataFeed(symbols, interval, client)
                    state = {
                        sym: {
                            "position": 0, "qty": 0.0, "entry_price": None, "entry_time": None,
                            "stop": None, "take": None, "realized_pnl": 0.0, "unrealized_pnl": 0.0
                        } for sym in symbols
                    }
                    data_feed.start()
                    start_time = time.time()
                    while time.time() - start_time < max_wait:
                        if all(sym in data_feed.data and not data_feed.data[sym].empty for sym in symbols):
                            break
                        logging.info("Waiting for WebSocket data to populate for fallback symbol...")
                        time.sleep(5)
                    if any(sym not in data_feed.data or data_feed.data[sym].empty for sym in symbols):
                        logging.error("Failed to get WebSocket data even for fallback symbol BTCUSDT")
                        sys.exit(1)
            except BinanceAPIException as e:
                logging.error(f"Default symbol BTCUSDT also invalid: {e}")
                sys.exit(1)
        else:
            logging.error("No WebSocket data available and force_default=False; exiting")
            sys.exit(1)

    # Paper mode dry run test orders
    if mode == "paper":
        test_symbol = symbols[0]
        symbol_info = client.get_symbol_info(test_symbol)
        min_notional = get_min_order(symbol_info)
        df_ind = fetch_candle_data(client, test_symbol, interval)
        price = float(df_ind['close'].iloc[-1])
        order_qty = max(0.001, min_notional / price)

        step_size = next((float(f['stepSize']) for f in symbol_info.get('filters', [])
                          if f['filterType'] == 'LOT_SIZE'), None)
        order_qty_str = format_quantity(order_qty, step_size) if step_size else str(order_qty)

        try:
            logging.debug(f"Placing test order: symbol={test_symbol}, side=BUY, qty={order_qty_str}")
            client.create_test_order(symbol=test_symbol, side='BUY', type='MARKET', quantity=order_qty_str)
            logging.info("Test order ok.")
        except BinanceAPIException as e:
            logging.error(f"Paper test order failed: {e}")

        try:
            test_qty = max(0.0001, min_notional / price)
            test_qty, _, _ = enforce_exchange_filters(client, test_symbol, test_qty, None)
            test_qty_str = format_quantity(test_qty, step_size) if step_size else str(test_qty)
            resp = place_market_order(client, test_symbol, "BUY", test_qty_str, mode)
            logging.info("Dry-run internal test order placed successfully.")
        except Exception as e:
            logging.error(f"Dry-run internal test order failed: {e}")

    poll_sleep = 60 if interval.endswith("m") else 5

    # Main trading loop
    while True:
        now = datetime.now(timezone.utc)
        if (now - day_start) > timedelta(days=1):
            day_start = now
            starting_day_balance = max(1.0, get_free_usdt(client))
            day_realized_pnl = 0.0
            logging.info("New day baseline reset.")

        current_prices = {
            sym: float(data_feed.data[sym].iloc[-1]['Close']) if sym in data_feed.data and not data_feed.data[sym].empty else 0.0
            for sym in symbols
        }

        portfolio_risk = portfolio_var(state, current_prices, data_feed.data)
        if portfolio_risk > abs(var_limit):
            send_alert("Portfolio VaR Limit Hit", f"VaR: {portfolio_risk:.2%}, stopping bot.")
            logging.warning("Portfolio VaR limit hit; stopping.")
            break

        for sym in symbols:
            try:
                if sym not in data_feed.data or data_feed.data[sym].empty:
                    logging.warning(f"{sym}: no WebSocket data; skipping")
                    continue
                df_ind = data_feed.data[sym]
                if 'ATR' not in df_ind.columns or df_ind['ATR'].isna().all():
                    df_ind = indicators(df_ind)
                    data_feed.data[sym] = df_ind

                row = df_ind.iloc[-1]
                atr = float(row['ATR'])
                price = float(row['Close'])

                if ml and not ml.trained and len(df_ind) > 200:
                    try:
                        ml.fit(df_ind.iloc[:-1])
                    except Exception as e:
                        logging.warning(f"ML training skipped: {e}")

                s = state[sym]

                if s["position"] != 0 and s["entry_price"] is not None and s["qty"] > 0:
                    side = "BUY" if s["position"] == 1 else "SELL"
                    s["unrealized_pnl"] = trade_pnl(side, s["entry_price"], price, s["qty"])

                if s["position"] != 0 and s["stop"] is not None and s["take"] is not None:
                    hit_stop = (price <= s["stop"]) if s["position"] == 1 else (price >= s["stop"])
                    hit_take = (price >= s["take"]) if s["position"] == 1 else (price <= s["take"])
                    if hit_stop or hit_take:
                        exit_side = "SELL" if s["position"] == 1 else "BUY"
                        adj_qty, _, _ = enforce_exchange_filters(client, sym, s["qty"], None)
                        step_size = next((float(f['stepSize']) for f in client.get_symbol_info(sym)['filters']
                                         if f['filterType'] == 'LOT_SIZE'), None)
                        qty_str = format_quantity(adj_qty, step_size) if step_size else str(adj_qty)
                        resp = place_market_order(client, sym, exit_side, qty_str, mode)

                        gross = trade_pnl("BUY" if s["position"] == 1 else "SELL", s["entry_price"], price, float(qty_str))
                        fees = (order_notional("BUY", s["entry_price"], float(qty_str)) +
                                order_notional(exit_side, price, float(qty_str))) * fee_rate
                        realized = gross - fees
                        s["realized"] += realized
                        day_realized_pnl += realized

                        event_body = {
                            "Event": "Trade Exit",
                            "Reason": "take" if hit_take else "stop",
                            "Symbol": sym,
                            "Side": exit_side,
                            "Quantity": s["qty"],
                            "Entry Price": s["entry_price"],
                            "Exit Price": price,
                            "Gross PnL": round(gross, 6),
                            "Fees": round(fees, 6),
                            "Net PnL": round(realized, 6),
                            "ATR": float(row['ATR']),
                            "RSI": float(row['RSI']),
                            "VWAP": float(row['VWAP']),
                            "Stoch_RSI_K": float(row['Stoch_RSI_K']),
                            "Stoch_RSI_D": float(row['Stoch_RSI_D']),
                            "Vol_Osc": float(row['Vol_Osc']),
                            "Order ID": resp.get("clientOrderId", "N/A"),
                            "Timestamp": now.isoformat()
                        }
                        logging.info(json.dumps(event_body))
                        send_alert(f"Trade Exit: {sym}", event_body)

                        if trade_log_csv:
                            append_trade_csv(trade_log_csv, {
                                "ts": now.isoformat(),
                                "symbol": sym,
                                "side": "BUY" if s["position"] == 1 else "SELL",
                                "qty": s["qty"],
                                "entry": s["entry_price"],
                                "exit": price,
                                "reason": event_body["Reason"],
                                "gross": gross,
                                "fees": fees,
                                "realized": realized
                            })

                        s.update({"position": 0, "qty": 0.0, "entry_price": None,
                                  "entry_time": None, "stop": None, "take": None,
                                  "unrealized_pnl": 0.0})

                if s["position"] == 0:
                    signal = simple_signal(row)
                    if ml and ml.trained:
                        ml_sig = ml.predict_signal(row)
                        if ml_sig is not None and ml_sig != 0:
                            signal = ml_sig

                    if signal != 0 and atr > 0:
                        free_usdt = max(0.0, get_free_usdt(client))
                        qty = position_size_from_atr(free_usdt, price, atr, risk_per_trade)
                        qty, adj_price, min_notional = enforce_exchange_filters(client, sym, qty, price)

                        if price * qty < max(min_notional, 5.0):
                            logging.info(f"{sym}: qty too small after filters; skipping (qty={qty})")
                        else:
                            adj_qty, _, _ = enforce_exchange_filters(client, sym, qty, None)
                            step_size = next((float(f['stepSize']) for f in client.get_symbol_info(sym)['filters']
                                             if f['filterType'] == 'LOT_SIZE'), None)
                            qty_str = format_quantity(adj_qty, step_size) if step_size else str(adj_qty)
                            side = "BUY" if signal == 1 else "SELL"
                            resp = place_market_order(client, sym, side, qty_str, mode)

                            stop = (price - sl_mult * atr) if signal == 1 else (price + sl_mult * atr)
                            take = (price + tp_mult * sl_mult * atr) if signal == 1 else (price - tp_mult * sl_mult * atr)

                            state[sym].update({
                                "position": 1 if signal == 1 else -1,
                                "qty": float(qty_str),
                                "entry_price": price,
                                "entry_time": now,
                                "stop": float(stop),
                                "take": float(take),
                                "unrealized_pnl": 0.0
                            })

                            event_body = {
                                "Event": "Trade Entry",
                                "Symbol": sym,
                                "Side": side,
                                "Quantity": qty,
                                "Entry Price": price,
                                "Stop Loss": float(stop),
                                "Take Profit": float(take),
                                "ATR": float(atr),
                                "RSI": float(row['RSI']),
                                "VWAP": float(row['VWAP']),
                                "Stoch_RSI_K": float(row['Stoch_RSI_K']),
                                "Stoch_RSI_D": float(row['Stoch_RSI_D']),
                                "Vol_Osc": float(row['Vol_Osc']),
                                "Order ID": resp.get("clientOrderId", "N/A"),
                                "Timestamp": now.isoformat()
                            }
                            logging.info(json.dumps(event_body))
                            send_alert(f"Trade Entry: {sym}", event_body)

                # Exit position if max hours exceeded
                if s["position"] != 0 and s["entry_time"] and (now - s["entry_time"]) > timedelta(hours=max_hours):
                    exit_side = "SELL" if s["position"] == 1 else "BUY"
                    adj_qty, _, _ = enforce_exchange_filters(client, sym, s["qty"], None)
                    step_size = next((float(f['stepSize']) for f in client.get_symbol_info(sym)['filters']
                                     if f['filterType'] == 'LOT_SIZE'), None)
                    qty_str = format_quantity(adj_qty, step_size) if step_size else str(adj_qty)

                    resp = place_market_order(client, sym, exit_side, qty_str, mode)

                    gross = trade_pnl("BUY" if s["position"] == 1 else "SELL", s["entry_price"], price, float(qty_str))
                    fees = (order_notional("BUY", s["entry_price"], float(qty_str)) +
                            order_notional(exit_side, price, float(qty_str))) * fee_rate
                    realized = gross - fees
                    s["realized"] += realized
                    day_realized_pnl += realized

                    event_body = {
                        "Event": "Trade Exit",
                        "Reason": "time",
                        "Symbol": sym,
                        "Side": exit_side,
                        "Quantity": s["qty"],
                        "Entry Price": s["entry_price"],
                        "Exit Price": price,
                        "Gross PnL": round(gross, 6),
                        "Fees": round(fees, 6),
                        "Net PnL": round(realized, 6),
                        "ATR": float(row['ATR']),
                        "RSI": float(row['RSI']),
                        "VWAP": float(row['VWAP']),
                        "Stoch_RSI_K": float(row['Stoch_RSI_K']),
                        "Stoch_RSI_D": float(row['Stoch_RSI_D']),
                        "Vol_Osc": float(row['Vol_Osc']),
                        "Order ID": resp.get("clientOrderId", "N/A"),
                        "Timestamp": now.isoformat()
                    }
                    logging.info(json.dumps(event_body))
                    send_alert(f"Trade Exit: {sym}", event_body)

                    if trade_log_csv:
                        append_trade_csv(trade_log_csv, {
                            "ts": now.isoformat(),
                            "symbol": sym,
                            "side": exit_side,
                            "qty": s["qty"],
                            "entry": s["entry_price"],
                            "exit": price,
                            "reason": "time",
                            "gross": gross,
                            "fees": fees,
                            "realized": realized
                        })

                    s.update({"position": 0, "qty": 0.0, "entry_price": None,
                              "entry_time": None, "stop": None, "take": None,
                              "unrealized_pnl": 0.0})

                if day_realized_pnl <= daily_loss_limit * starting_day_balance:
                    send_alert("Daily Loss Limit Hit", f"Loss: {day_realized_pnl:.2f} USDT, stopping bot.")
                    logging.warning("Daily loss limit reached; stopping bot safely.")
                    return

                # Update Prometheus metrics
                total_equity = starting_day_balance + day_realized_pnl + sum(s["unrealized_pnl"] for s in state.values())
                equity_gauge.set(total_equity)
                drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0
                drawdown_gauge.set(drawdown * 100)
                if drawdown > 0.5:
                    send_alert("Max Drawdown Alert", f"Drawdown exceeded 50%: {drawdown:.2%}")
                    logging.warning("Max drawdown exceeded; stopping bot.")
                    return
                peak_equity = max(peak_equity, total_equity)

            except Exception as e:
                logging.exception(f"{sym}: loop error: {e}")

        time.sleep(poll_sleep)


# -------- Backtest --------
def backtest(symbol: str, interval: str, days: int = 60, risk_per_trade: float = 0.01,
             fee_rate: float = 0.001, tp_mult: float = 3.0, sl_mult: float = 2.0,
             use_ml: bool = True):
    setup_logging(None)
    api_key = os.getenv("BINANCE_KEY", "x")
    api_secret = os.getenv("BINANCE_SECRET", "x")
    client = SafeBinanceClient(api_key, api_secret, testnet=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = fetch_klines_series(client, symbol, interval, start, end)
    if df.empty:
        logging.error("No data for backtest")
        return

    df_ind = indicators(df).dropna().copy()
    ml = RollingML() if use_ml else None
    if ml and len(df_ind) > 200 and _HAS_SK:
        ml.fit(df_ind.iloc[:-50])

    balance = 10000.0
    equity_curve = []
    peak = balance
    max_dd = 0.0

    pos = 0
    qty = 0.0
    entry = None
    stop = None
    take = None

    trades = 0
    wins = 0
    losses = 0
    gross_pnl_sum = 0.0
    fee_sum = 0.0
    realized_sum = 0.0

    def size_for(price, atr):
        return position_size_from_atr(balance, price, atr, risk_per_trade)

    for i in range(len(df_ind) - 1):
        row = df_ind.iloc[i]
        nxt = df_ind.iloc[i + 1]
        price = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])
        atr = float(row['ATR'])

        if pos != 0 and stop is not None and take is not None:
            hit_stop = (low <= stop) if pos == 1 else (high >= stop)
            hit_take = (high >= take) if pos == 1 else (low <= take)

            if hit_stop or hit_take:
                exit_price = stop if hit_stop else take
                side_in = "BUY" if pos == 1 else "SELL"
                exit_side = "SELL" if pos == 1 else "BUY"
                gross = trade_pnl(side_in, entry, exit_price, qty)
                fees = order_notional(side_in, entry, qty) * fee_rate + order_notional(exit_side, exit_price, qty) * fee_rate
                realized = gross - fees
                balance += realized

                trades += 1
                gross_pnl_sum += gross
                fee_sum += fees
                realized_sum += realized
                if realized > 0:
                    wins += 1
                else:
                    losses += 1

                pos = 0
                qty = 0.0
                entry = stop = take = None

        if pos == 0 and atr > 0:
            sig = simple_signal(row)
            if ml and ml.trained:
                ml_sig = ml.predict_signal(row)
                if ml_sig is not None and ml_sig != 0:
                    sig = ml_sig
            if sig != 0:
                q = size_for(price, atr)
                if q > 0:
                    pos = 1 if sig == 1 else -1
                    qty = q
                    entry = price
                    stop = (price - sl_mult * atr) if pos == 1 else (price + sl_mult * atr)
                    take = (price + sl_mult * tp_mult * atr) if pos == 1 else (price - sl_mult * tp_mult * atr)
                    fee = order_notional("BUY" if pos == 1 else "SELL", entry, qty) * fee_rate
                    balance -= fee
                    fee_sum += fee

        mtm = 0.0
        if pos != 0 and entry is not None and qty > 0:
            side_in = "BUY" if pos == 1 else "SELL"
            mtm = trade_pnl(side_in, entry, price, qty)
        equity = balance + mtm
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    if pos != 0 and entry is not None and qty > 0:
        final_price = float(df_ind.iloc[-1]['Close'])
        side_in = "BUY" if pos == 1 else "SELL"
        exit_side = "SELL" if pos == 1 else "BUY"
        gross = trade_pnl(side_in, entry, final_price, qty)
        fees = order_notional(exit_side, final_price, qty) * fee_rate
        realized = gross - fees
        balance += realized
        trades += 1
        gross_pnl_sum += gross
        fee_sum += fees
        realized_sum = realized

    start_equity = 10000.0
    end_equity = balance
    ret = (end_equity / start_equity) - 1.0
    win_rate = (wins / trades) if trades > 0 else 0.0

    logging.info(json.dumps({
        "event": "backtest_summary",
        "symbol": symbol,
        "interval": interval,
        "bars": len(df_ind),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(100 * win_rate, 2),
        "gross_pnl": round(gross_pnl_sum, 2),
        "fees": round(fee_sum, 2),
        "realized_pnl": round(realized_sum, 2),
        "start_equity": start_equity,
        "end_equity": round(end_equity, 2),
        "return_%": round(100 * ret, 2),
        "max_drawdown_%": round(100 * max_dd, 2)
    }))

# -------- CLI --------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="TradeBot Pro")
    p.add_argument("--symbols", nargs="+", default=[], help="Symbols to trade")
    p.add_argument("--auto-select", choices=["top_gainer", "top_loser", "both"], help="Auto-select trading symbols")
    p.add_argument("--min-volume", type=float, default=5e7, help="Min 24h USDT volume for auto-select")
    p.add_argument("--interval", default="5m", help="Kline interval (e.g., 1m,5m,1h)")
    p.add_argument("--mode", choices=["paper", "live", "backtest"], default=os.getenv("MODE", "paper"))
    p.add_argument("--days", type=int, default=60, help="Days of history for backtest or features")
    p.add_argument("--risk", type=float, default=float(os.getenv("RISK_PER_TRADE", "0.01")),
                   help="Risk per trade fraction")
    p.add_argument("--daily_loss_limit", type=float, default=float(os.getenv("DAILY_LOSS_LIMIT", "-0.05")),
                   help="Daily loss limit as fraction of starting-day balance")
    p.add_argument("--var_limit", type=float, default=float(os.getenv("PORTFOLIO_VAR_LIMIT", "-0.05")),
                   help="Portfolio VaR limit as fraction")
    p.add_argument("--max_hours", type=int, default=int(os.getenv("MAX_TRADE_HOURS", "24")),
                   help="Max hours to keep position open")
    p.add_argument("--log_file", default=os.getenv("LOG_FILE", "trader.log"), help="Optional file log path")
    p.add_argument("--trade_log_csv", default=os.getenv("TRADE_LOG_CSV", ""), help="Optional CSV trade log path")
    p.add_argument("--use_testnet", action="store_true" if getenv_bool("USE_TESTNET", True) else "store_false",
                   help="Use Binance testnet (paper mode)")
    p.add_argument("--force-default", action="store_true" if getenv_bool("FORCE_DEFAULT", True) else "store_false",
                   help="Fallback to BTCUSDT if no valid symbols (default: true)")
    p.add_argument("--no-ml", dest="use_ml", action="store_false", help="Disable ML blending")
    p.add_argument("--fee", type=float, default=float(os.getenv("FEE_RATE", "0.001")), help="Fee rate (e.g., 0.001)")
    p.add_argument("--tp_mult", type=float, default=float(os.getenv("TP_MULT", "3.0")),
                   help="Take-profit multiple (of SL move)")
    p.add_argument("--sl_mult", type=float, default=float(os.getenv("SL_MULT", "2.0")),
                   help="Stop-loss multiple (ATR)")
    p.set_defaults(use_ml=True)
    return p.parse_args()

def main():
    args = parse_args()

    trade_log_csv = args.trade_log_csv if args.trade_log_csv else None

    if args.mode == "backtest":
        symbols = args.symbols if args.symbols else ["BTCUSDT"]
        if len(symbols) != 1:
            logging.info("Backtest supports one symbol at a time. Taking first symbol.")
        backtest(symbols[0], args.interval, args.days, args.risk,
                 fee_rate=args.fee, tp_mult=args.tp_mult, sl_mult=args.sl_mult,
                 use_ml=args.use_ml)
    else:
        symbols = args.symbols
        run_live_or_paper(
            symbols=symbols,
            interval=args.interval,
            mode=args.mode,
            risk_per_trade=args.risk,
            daily_loss_limit=args.daily_loss_limit,
            var_limit=args.var_limit,
            max_hours=args.max_hours,
            use_testnet=args.use_testnet,
            log_file=args.log_file if args.log_file else None,
            use_ml=args.use_ml,
            auto_select=args.auto_select,
            min_volume=args.min_volume,
            fee_rate=args.fee,
            tp_mult=args.tp_mult,
            sl_mult=args.sl_mult,
            trade_log_csv=trade_log_csv,
            force_default=args.force_default
        )

if __name__ == "__main__":
    main()