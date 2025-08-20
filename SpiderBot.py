import requests.exceptions
from decimal import Decimal, ROUND_DOWN
from math import floor
import decimal
from decimal import ROUND_DOWN, Decimal
import logging
import time
import pandas as pd
import traceback
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta, timezone
from binance.exceptions import BinanceAPIException
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(filename="trader_longshort.log",
                    level=logging.INFO, format="%(asctime)s %(message)s")


class BinanceClient:
    def __init__(self, api_key, secret_key):
        self.client = Client(api_key, secret_key)

    def get_account_info(self):
        """Fetch account balance from Binance."""
        try:
            return self.client.get_account()
        except BinanceAPIException as e:
            logging.error(f"Error fetching account info: {str(e)}")
            return None

    def get_symbol_info(self, symbol):
        """Fetch symbol information from Binance."""
        try:
            return self.client.get_symbol_info(symbol)
        except BinanceAPIException as e:
            logging.error(f"Error fetching symbol info for {symbol}: {str(e)}")
            return None

    def create_order(self, symbol, side, order_type, quantity):
        """Place a new market order."""
        try:
            return self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
        except BinanceAPIException as e:
            logging.error(f"Error placing order: {str(e)}")
            print((f"Error placing order: {str(e)}"))
            return None

    def get_historical_klines(self, symbol, interval, start_time, end_time=None, limit=500):
        """Fetch historical candlestick data (Klines) from Binance."""
        try:
            return self.client.get_klines(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time, limit=limit)
        except BinanceAPIException as e:
            logging.error(f"Error fetching historical klines: {str(e)}")
            return None


class TraderLongShort:

    def __init__(self, client, symbols, bar_length, return_thresh, volume_thresh, units, atr_multiplier=2, take_profit_ratio=3):
        self.symbols = symbols  # Handling multiple assets
        self.bar_length = bar_length
        self.units = units
        self.trades = 0
        self.trade_values = []
        self.cum_profits = 0
        # Track positions for each symbol
        self.position = {symbol: 0 for symbol in symbols}
        self.atr_window = 14
        self.return_thresh = return_thresh
        self.volume_thresh = volume_thresh
        self.atr_multiplier = atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        # Entry time for each symbol
        self.entry_time = {symbol: None for symbol in symbols}
        self.client = client
        # Adding a simple Logistic Regression model
        self.ml_model = LogisticRegression()

    def fetch_historical_data(self, symbol, start_time, end_time, interval="5m"):
        """Fetch historical kline data from Binance."""
        try:
            start_time_ms = int(start_time.timestamp() * 1000)
            end_time_ms = int(end_time.timestamp() * 1000)
            response = self.client.get_historical_klines(
                symbol=symbol, interval=interval, start_time=start_time_ms, end_time=end_time_ms)
            response = [entry[:11] for entry in response]
            df = pd.DataFrame(response)
            df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                          "Close Time", "Quote Asset Volume", "Number of Trades",
                          "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume"]
            df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
            df.set_index("Date", inplace=True)
            for column in ["Open", "High", "Low", "Close", "Volume"]:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
            logging.info(f"Fetched {len(df)} bars of {
                         interval} data for {symbol}.")
            return df
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def fetch_historical_data_1h(self, symbol, days):
        """Fetch 1-hour interval data for a given symbol for the last `days` days."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Fetch historical data for the given symbol
        df_1h = self.fetch_historical_data(
            symbol, start_time=start_time, end_time=end_time, interval='1h')

        if df_1h is None or df_1h.empty:
            logging.error(f"No 1-hour data fetched for {symbol}.")
            return None

        return df_1h

    def calculate_indicators(self, df):
        """Calculate ATR, RSI, ADX, MACD, Bollinger Bands."""
        df["ATR"] = self.calculate_atr(df, self.atr_window)
        df["RSI"] = self.calculate_rsi(df["Close"], window=14)
        df["ADX"] = self.calculate_adx(df)
        df["MACD"], df["Signal"] = self.calculate_macd(df["Close"])
        df["Upper Band"], df["Lower Band"] = self.calculate_bollinger_bands(
            df["Close"])
        return df

    def calculate_atr(self, df, window):
        """Calculate ATR."""
        df["high_low"] = df["High"] - df["Low"]
        df["high_close"] = abs(df["High"] - df["Close"].shift())
        df["low_close"] = abs(df["Low"] - df["Close"].shift())
        df["true_range"] = df[["high_low",
                               "high_close", "low_close"]].max(axis=1)
        df["ATR"] = df["true_range"].rolling(
            window=window, min_periods=1).mean()
        return df["ATR"]

    def calculate_rsi(self, series, window):
        """Calculate RSI."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_adx(self, df, window=14):
        """Calculate ADX."""
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(
            abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
        df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                             df['High'] - df['High'].shift(), 0)
        df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                             df['Low'].shift() - df['Low'], 0)
        df['+DI'] = 100 * (df['+DM'].rolling(window).sum() /
                           df['TR'].rolling(window).sum())
        df['-DI'] = 100 * (df['-DM'].rolling(window).sum() /
                           df['TR'].rolling(window).sum())
        df['DX'] = 100 * np.abs((df['+DI'] - df['-DI']) /
                                (df['+DI'] + df['-DI']))
        df['ADX'] = df['DX'].rolling(window).mean()
        return df['ADX']

    def calculate_macd(self, close_prices, short_window=12, long_window=26, signal_window=9):
        exp1 = close_prices.ewm(span=short_window, adjust=False).mean()
        exp2 = close_prices.ewm(span=long_window, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def calculate_bollinger_bands(self, close_prices, window=20, num_std_dev=2):
        rolling_mean = close_prices.rolling(window=window).mean()
        rolling_std = close_prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return upper_band, lower_band

    def adjust_position_size(self, atr_value, symbol):
        """Adjust position size based on ATR and Binance's minimum size requirements."""
        balance_info = self.client.get_account_info()
        usdc_balance = next(
            (item for item in balance_info['balances'] if item['asset'] == 'USDC'), None)
        if usdc_balance is None:
            raise ValueError("USDC balance not found")
        balance = float(usdc_balance['free'])
        risk_per_trade = 0.20
        position_size = (balance * risk_per_trade) / atr_value

        # Fetch minimum order size for the symbol
        min_order_size = self.get_min_order_size(symbol)
        logging.info(f"Position size calculated: {
                     position_size}, Minimum order size: {min_order_size}")

        # Ensure position size is greater than the minimum order size
        return max(min(position_size, balance), min_order_size)

    def get_min_order_size(self, symbol):
        """Fetch the minimum order size for a given symbol from Binance exchange info."""
        try:
            exchange_info = client.get_symbol_info(symbol)
            filters = exchange_info.get("filters", [])
            for f in filters:
                if f['filterType'] == 'LOT_SIZE':
                    return float(f['minQty'])
        except Exception as e:
            logging.error(f"Error fetching min order size for {
                          symbol}: {str(e)}")
            return 0.001  # Default to a small size if there's an error

    def dynamic_position_sizing(self, atr_value, volatility, symbol, current_price, retries=3, delay=2):
        """
        Adjust position size based on ATR and current market volatility.
        Higher volatility leads to smaller positions; lower volatility allows larger positions.
        Adds retries and detailed logging.
        """
        balance_info = self.client.get_account_info()
        usdc_balance = next(
            (item for item in balance_info['balances'] if item['asset'] == 'USDC'), None)

        if usdc_balance is None:
            logging.error(f"USDC balance not found in the API response")
            return None

        balance = float(usdc_balance['free'])

        # Risk per trade (e.g., 1-2% of balance)
        risk_per_trade = 0.02

        # Add a small value to volatility and ATR to prevent division by zero or too small values
        atr_value = max(atr_value, 1e-5)
        volatility = max(volatility, 1e-5)

        # Position size is inversely related to volatility
        position_size = (balance * risk_per_trade) / (atr_value * volatility)

        # Retry logic for fetching symbol info
        for attempt in range(retries):
            try:
                symbol_info = self.client.get_symbol_info(symbol)
                if symbol_info:
                    logging.info(f"Symbol info for {symbol}: {symbol_info}")
                    lot_size_filter = next(
                        f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
                    min_order_size = float(lot_size_filter['minQty'])
                    step_size = float(lot_size_filter['stepSize'])

                    # Truncate position size to match the step size
                    position_size = float(
                        self.truncate_to_step_size(position_size, step_size))

                    # Ensure the notional value (price * position size) meets the exchange's requirement
                    notional_value = current_price * position_size
                    min_notional_filter = next(
                        f for f in symbol_info['filters'] if f['filterType'] == 'NOTIONAL')
                    min_notional = float(min_notional_filter['minNotional'])

                    # Adjust position size if notional value is below the minimum required
                    if notional_value < min_notional:
                        logging.warning(f"Notional value {notional_value:.8f} is less than the minimum notional {
                                        min_notional:.8f}. Adjusting position size.")
                        # Recalculate position size to meet the minimum notional requirement
                        position_size = min_notional / current_price
                        position_size = float(
                            self.truncate_to_step_size(position_size, step_size))
                        logging.info(f"Adjusted position size to meet the minimum notional: {
                            position_size:.8f}")

                    # If the adjusted position size is still below the minimum order size, skip the trade
                    if position_size < min_order_size:
                        logging.warning(f"Position size {position_size:.8f} is less than the minimum order size {
                                        min_order_size:.8f}. Skipping trade.")
                        return None

                    logging.info(f"Dynamic position sizing for {symbol}: Calculated: {position_size:.8f}, Minimum required: {
                        min_order_size:.8f}, Step size: {step_size:.8f}, Notional: {notional_value:.8f}")
                    return position_size
                else:
                    logging.error(f"Could not fetch symbol info for {
                        symbol}. Response: {symbol_info}")
                    raise ValueError(f"Symbol info not found for {symbol}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching symbol info for {symbol}: {e}. Attempt {
                    attempt + 1} of {retries}. Retrying in {delay} seconds.")
                time.sleep(delay)
            except Exception as e:
                logging.error(
                    f"Unexpected error during position size calculation for {symbol}: {e}")
                break  # No point in retrying if it's not an API issue

        logging.error(f"Failed to fetch symbol info for {
            symbol} after {retries} attempts.")
        return None

    def truncate_to_step_size(self, quantity, step_size):
        """
        Truncate the quantity to match the required step size.
        Binance requires that the quantity matches the 'stepSize' defined in the LOT_SIZE filter.
        """
        step_size_decimal = Decimal(str(step_size))
        quantity_decimal = Decimal(str(quantity))

        # Truncate the quantity to the appropriate step size
        truncated_quantity = quantity_decimal.quantize(
            step_size_decimal, rounding=ROUND_DOWN)

        # Return the truncated quantity as a float
        return float(truncated_quantity)

    def backtest(self, days=90):
        """Backtest function with MACD and Bollinger Bands for multiple symbols."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Dictionary to hold data for each symbol
        all_data = {}

        # Loop through each symbol
        for symbol in self.symbols:
            logging.info(f"Backtesting for symbol: {symbol}")

            # Fetch historical data for the current symbol
            df = self.fetch_historical_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval=self.bar_length
            )

            if df is None or df.empty:
                logging.error(f"No data fetched for {symbol}")
                continue  # Skip to the next symbol if there's no data

            # Existing ATR, RSI, and ADX calculations for the fetched symbol's data
            df["returns"] = np.log(df.Close / df.Close.shift())
            df["ATR"] = self.calculate_atr(df, self.atr_window)
            df["MA_100"] = df["Close"].rolling(window=100).mean()
            df["RSI"] = self.calculate_rsi(df["Close"], window=14)
            df["ADX"] = self.calculate_adx(df)

            # Calculate MACD and Bollinger Bands for the current symbol
            df["MACD"], df["Signal"] = self.calculate_macd(df["Close"])
            df["Upper Band"], df["Lower Band"] = self.calculate_bollinger_bands(
                df["Close"])

            # Fetch the 1-hour data for trend confirmation
            df_1h = self.fetch_historical_data_1h(symbol, days=90)

            if df_1h is None or df_1h.empty:
                logging.error(f"No 1-hour data fetched for {symbol}")
                continue

            df_1h["EMA_100"] = df_1h["Close"].ewm(
                span=100, adjust=False).mean()

            # Align the 1-hour data to match the 5-minute data
            df_1h_aligned = df_1h.reindex(df.index, method='nearest')

            # ATR volatility threshold
            volatility_threshold = df["ATR"].mean() * 2

            for i in range(1, len(df)):
                current_price = df["Close"].iloc[i]
                atr_value = df["ATR"].iloc[i]
                current_time = df.index[i]

                stop_loss = current_price - (self.atr_multiplier * atr_value)
                take_profit = current_price + \
                    (self.take_profit_ratio * (current_price - stop_loss))

                # Long term trend check using 1-hour EMA
                long_term_trend_up = df_1h_aligned["EMA_100"].iloc[i] < current_price
                long_term_trend_down = df_1h_aligned["EMA_100"].iloc[i] > current_price

                # Pass current_price to the dynamic_position_sizing method
                position_size = self.dynamic_position_sizing(
                    atr_value, volatility_threshold, symbol, current_price)

                if not position_size:
                    logging.error(f"Unable to determine position size for {
                        symbol} at {current_time}")
                    continue  # Skip trade if position size is invalid

                trending_market = df["ADX"].iloc[i] > 20

                # MACD and Signal conditions
                macd_cross_up = df["MACD"].iloc[i] > df["Signal"].iloc[i]
                macd_cross_down = df["MACD"].iloc[i] < df["Signal"].iloc[i]

                # Bollinger Band conditions
                price_near_lower_band = current_price <= df["Lower Band"].iloc[i]
                price_near_upper_band = current_price >= df["Upper Band"].iloc[i]

                # Long and short conditions
                long_condition = (trending_market and macd_cross_up and price_near_lower_band and (
                    df["RSI"].iloc[i] < 35) and long_term_trend_up)
                short_condition = (trending_market and macd_cross_down and price_near_upper_band and (
                    df["RSI"].iloc[i] > 65) and long_term_trend_down)

                # Check if conditions are met and initiate trades
                if atr_value > volatility_threshold:
                    if long_condition and self.position[symbol] == 0:
                        self.execute_trade(
                            "BUY", symbol, current_price, position_size, stop_loss, take_profit)
                        self.entry_time[symbol] = current_time
                        self.position[symbol] = 1
                    elif short_condition and self.position[symbol] == 0:
                        self.execute_trade(
                            "SELL", symbol, current_price, position_size, stop_loss, take_profit)
                        self.entry_time[symbol] = current_time
                        self.position[symbol] = -1
                    elif self.position[symbol] == 1 and current_price <= stop_loss:
                        self.execute_trade(
                            "SELL", symbol, current_price, position_size, stop_loss, take_profit, exit=True)
                        self.position[symbol] = 0
                    elif self.position[symbol] == 1 and current_price >= take_profit:
                        self.execute_trade(
                            "SELL", symbol, current_price, position_size, stop_loss, take_profit, exit=True)
                        self.position[symbol] = 0
                    elif self.position[symbol] == -1 and current_price >= stop_loss:
                        self.execute_trade(
                            "BUY", symbol, current_price, position_size, stop_loss, take_profit, exit=True)
                        self.position[symbol] = 0
                    elif self.position[symbol] == -1 and current_price <= take_profit:
                        self.execute_trade(
                            "BUY", symbol, current_price, position_size, stop_loss, take_profit, exit=True)
                        self.position[symbol] = 0

                # Exit trade if it has been open for more than 24 hours
                if self.entry_time[symbol] and (current_time - self.entry_time[symbol] > pd.Timedelta(hours=24)) and self.position[symbol] != 0:
                    logging.info(f"Exiting trade after 24 hours: Current Price: {
                        current_price}")
                    self.execute_trade("SELL" if self.position[symbol] == 1 else "BUY", symbol,
                                       current_price, position_size, stop_loss, take_profit, exit=True)
                    self.position[symbol] = 0

            # Store the dataframe for this symbol in the dictionary
            all_data[symbol] = df

        logging.info(f"Backtesting complete for all symbols.")
        return all_data

    def execute_trade(self, side, symbol, price, position_size, stop_loss, take_profit, exit=False):
        """Execute a trade and log it."""
        try:
            logging.info(f"Executing {side} trade for {
                symbol} with position size: {position_size}")
            order = self.client.create_order(
                symbol=symbol, side=side, order_type='MARKET', quantity=position_size)
            logging.info(f"Order created for {symbol}: {order}")
            self.trades += 1
            logging.info(f"Trade {self.trades}: {side} at {price} for {
                symbol}. Position size: {position_size}, SL: {stop_loss}, TP: {take_profit}")
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}")
            logging.error(traceback.format_exc())


if __name__ == "__main__":

    api_key = ""
    secret_key = ""

    client = BinanceClient(api_key=api_key, secret_key=secret_key)
    info = client.get_account_info()

    # Find and print the USDC balance
    usdc_balance = next(
        (item for item in info['balances'] if item['asset'] == 'USDC'), None)

    if usdc_balance:
        print(f"USDC Balance: {usdc_balance['free']}")
    else:
        print("USDC balance not found in the account.")

    # Example symbols to trade
    symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC"]
    bar_length = "5m"
    return_thresh = [-0.0001, 0.0001]
    volume_thresh = [-3, 3]
    units = 0.01

    trader = TraderLongShort(client=client, symbols=symbols, bar_length=bar_length,
                             return_thresh=return_thresh, volume_thresh=volume_thresh, units=units)
    trader.backtest(days=90)

    # Set up trading periods for logging profit/loss
    start_time = datetime.now(timezone.utc)
    one_hour_timer = start_time
    twenty_four_hour_timer = start_time
    hourly_profit = 0
    daily_profit = 0

    while True:
        trader.backtest()

        # Log profit every hour
        current_time = datetime.now(timezone.utc)
        if current_time - one_hour_timer >= timedelta(hours=1):
            logging.info(
                f"1-Hour Profits: {trader.cum_profits - hourly_profit}")
            hourly_profit = trader.cum_profits  # Reset hourly profit counter
            one_hour_timer = current_time  # Reset timer for 1 hour

        # Log profit every 24 hours
        if current_time - twenty_four_hour_timer >= timedelta(hours=24):
            logging.info(
                f"24-Hour Profits: {trader.cum_profits - daily_profit}")
            daily_profit = trader.cum_profits  # Reset daily profit counter
            twenty_four_hour_timer = current_time  # Reset timer for 24 hours

        time.sleep(300)
