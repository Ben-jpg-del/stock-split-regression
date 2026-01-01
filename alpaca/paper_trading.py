import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import schedule

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from polygon import RESTClient as PolygonClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    quantity: int
    entry_time: datetime
    close_time: datetime
    entry_price: float
    closed: bool = False
    exit_price: Optional[float] = None

    def should_close(self, current_time: datetime) -> bool:
        return not self.closed and current_time >= self.close_time

    def on_split_occurred(self, split_factor: float) -> None:
        self.quantity = int(self.quantity / split_factor)


@dataclass
class SplitEvent:
    symbol: str
    execution_date: datetime
    split_factor: float  
    announced_date: Optional[datetime] = None


class SplitTradingBot:
    def __init__(
        self,
        alpaca_api_key: str,
        alpaca_secret_key: str,
        polygon_api_key: str,
        paper: bool = True,
        cash: float = 1000.0,
        max_open_trades: int = 4,
        hold_duration_days: int = 3,
        training_lookback_years: int = 4,
        min_order_value: float = 50.0,
        dry_run: bool = False
    ):
        self.trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        self.polygon_client = PolygonClient(polygon_api_key)

        self.cash = cash
        self.max_open_trades = max_open_trades
        self.hold_duration = timedelta(days=hold_duration_days)
        self.training_lookback = timedelta(days=training_lookback_years * 365)
        self.min_order_value = min_order_value
        self.dry_run = dry_run
        self.target_exposure_per_trade = 1.0 / max_open_trades

        self.model = LinearRegression()
        self.open_trades: Dict[str, List[Trade]] = {}
        self.sector_roc_history: pd.Series = pd.Series(dtype=float)
        self.tech_universe: Set[str] = set()
        self.processed_splits: Set[str] = set()  
        self.processed_occurred_splits: Set[str] = set()  
        self.sector_etf = "XLK"
        self.roc_period = 22
        self.resolution = TimeFrame.Hour  

        logger.info(f"Initialized SplitTradingBot (paper={paper}, dry_run={dry_run})")
        logger.info(f"Parameters: cash=${cash}, max_trades={max_open_trades}, "
                   f"hold_days={hold_duration_days}, lookback_years={training_lookback_years}")

    def get_account_info(self) -> dict:
        account = self.trading_client.get_account()
        return {
            'cash': float(account.cash),
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value)
        }

    def get_tech_universe(self) -> List[str]:

        tech_sic_ranges = [
            (3570, 3579),  # Computer and Office Equipment
            (3600, 3699),  # Electronic and Electrical Equipment
            (3810, 3849),  # Measuring and Analyzing Instruments
            (4810, 4899),  # Communications
            (7370, 7379),  # Computer Programming, Data Processing
            (7380, 7389),  # Misc Business Services (tech consulting)
        ]

        def is_tech_sic(sic_code: int) -> bool:
            for start, end in tech_sic_ranges:
                if start <= sic_code <= end:
                    return True
            return False

        try:
            tech_stocks = set()
            logger.info("Fetching dynamic tech universe from Polygon...")

            # Paginate through all active US stocks
            all_tickers = []
            for ticker in self.polygon_client.list_tickers(
                market='stocks',
                type='CS',  # Common Stock
                active=True,
                limit=1000
            ):
                all_tickers.append(ticker.ticker)

            logger.info(f"Found {len(all_tickers)} total active stocks, filtering by SIC code...")

            # Batch process to get SIC codes (with rate limiting consideration)
            processed = 0
            for symbol in all_tickers:
                try:
                    details = self.polygon_client.get_ticker_details(symbol)

                    if details and hasattr(details, 'sic_code') and details.sic_code:
                        if is_tech_sic(int(details.sic_code)):
                            tech_stocks.add(symbol)

                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed}/{len(all_tickers)} tickers, "
                                   f"found {len(tech_stocks)} tech stocks so far...")
                        time.sleep(0.1)  # Rate limiting

                except Exception:
                    continue

            if tech_stocks:
                self.tech_universe = tech_stocks
                logger.info(f"Tech universe: {len(self.tech_universe)} stocks (dynamically fetched via SIC codes)")
                return list(self.tech_universe)
            else:
                raise ValueError("No tech stocks found via SIC code filtering")

        except Exception as e:
            logger.warning(f"Dynamic SIC fetch failed ({e}), trying sector-based approach...")

            try:
                tech_stocks = set()

                for ticker in self.polygon_client.list_tickers(
                    market='stocks',
                    exchange='XNAS',  
                    type='CS',
                    active=True,
                    limit=1000
                ):
                    tech_stocks.add(ticker.ticker)

                if len(tech_stocks) >= 50:
                    self.tech_universe = tech_stocks
                    logger.info(f"Tech universe: {len(self.tech_universe)} stocks (NASDAQ fallback)")
                    return list(self.tech_universe)

            except Exception as fallback_error:
                logger.error(f"NASDAQ fallback also failed: {fallback_error}")

            logger.warning("Using minimal fallback tech universe")
            self.tech_universe = {
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'AMZN',
                'CRM', 'ORCL', 'ADBE', 'INTC', 'CSCO', 'IBM', 'NOW', 'INTU'
            }
            return list(self.tech_universe)

    def get_historical_splits(self, symbols: List[str] = None) -> List[SplitEvent]:
        if symbols is None:
            symbols = list(self.tech_universe)

        splits = []
        start_date = (datetime.now() - self.training_lookback).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            for split in self.polygon_client.list_splits(
                execution_date_gte=start_date,
                execution_date_lte=end_date,
                limit=10000  
            ):
                if split.ticker in symbols or not symbols:
                    split_factor = split.split_to / split.split_from
                    splits.append(SplitEvent(
                        symbol=split.ticker,
                        execution_date=datetime.strptime(split.execution_date, '%Y-%m-%d'),
                        split_factor=split_factor
                    ))

            logger.info(f"Found {len(splits)} historical splits (full dataset)")
            return splits

        except Exception as e:
            logger.error(f"Error getting historical splits: {e}")
            return []

    def get_upcoming_splits(self, days_ahead: int = 14) -> List[SplitEvent]:
        upcoming = []
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        try:
            for split in self.polygon_client.list_splits(
                execution_date_gte=start_date,
                execution_date_lte=end_date,
                limit=100
            ):
                if split.ticker in self.tech_universe:
                    split_factor = split.split_to / split.split_from
                    upcoming.append(SplitEvent(
                        symbol=split.ticker,
                        execution_date=datetime.strptime(split.execution_date, '%Y-%m-%d'),
                        split_factor=split_factor
                    ))

            logger.info(f"Found {len(upcoming)} upcoming splits in tech universe")
            return upcoming

        except Exception as e:
            logger.error(f"Error getting upcoming splits: {e}")
            return []

    def get_historical_prices(self, symbols: List[str], start: datetime, end: datetime,
                               resolution: TimeFrame = TimeFrame.Hour) -> pd.DataFrame:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=resolution,
                start=start,
                end=end
            )
            bars = self.data_client.get_stock_bars(request)

            data = {}
            for symbol in symbols:
                if symbol in bars.data:
                    symbol_bars = bars.data[symbol]
                    data[symbol] = {bar.timestamp: bar.open for bar in symbol_bars}

            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

        except Exception as e:
            logger.error(f"Error getting historical prices: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,  
                start=datetime.now() - timedelta(days=2),
                end=datetime.now()
            )
            bars = self.data_client.get_stock_bars(request)
            if symbol in bars.data and bars.data[symbol]:
                return bars.data[symbol][-1].close
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def calculate_sector_roc(self) -> float:
        try:
            end = datetime.now()
            start = end - timedelta(days=self.roc_period + 30)  

            request = StockBarsRequest(
                symbol_or_symbols=self.sector_etf,
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            bars = self.data_client.get_stock_bars(request)

            if self.sector_etf not in bars.data:
                logger.warning(f"No data for {self.sector_etf}")
                return 0.0

            closes = [bar.close for bar in bars.data[self.sector_etf]]

            if len(closes) < self.roc_period + 1:
                logger.warning(f"Insufficient data for ROC calculation")
                return 0.0

            current = closes[-1]
            past = closes[-(self.roc_period + 1)]
            roc = ((current - past) / past) * 100

            self.sector_roc_history.loc[datetime.now()] = roc
            cutoff = datetime.now() - self.training_lookback
            self.sector_roc_history = self.sector_roc_history[
                self.sector_roc_history.index > cutoff
            ]

            logger.info(f"Sector ROC ({self.sector_etf}): {roc:.2f}%")
            return roc

        except Exception as e:
            logger.error(f"Error calculating sector ROC: {e}")
            return 0.0

    def train_model(self) -> bool:
        logger.info("Training model on historical split data...")

        try:
            if not self.tech_universe:
                self.get_tech_universe()

            splits = self.get_historical_splits()
            if not splits:
                logger.warning("No historical splits found for training")
                return False

            split_symbols = list(set(s.symbol for s in splits))

            start = datetime.now() - self.training_lookback
            end = datetime.now()
            prices = self.get_historical_prices(split_symbols, start, end, TimeFrame.Day)

            if prices.empty:
                logger.warning("No price data available for training")
                return False

            self.calculate_sector_roc()  

            samples = []
            for split in splits:
                if split.symbol not in prices.columns:
                    continue

                split_date = split.execution_date
                entry_dates = prices.index[prices.index > split_date]
                if len(entry_dates) == 0:
                    continue
                entry_date = entry_dates[0]

                if pd.isna(prices.loc[entry_date, split.symbol]):
                    continue
                entry_price = prices.loc[entry_date, split.symbol]

                exit_target = entry_date + self.hold_duration
                exit_dates = prices.index[prices.index > exit_target]
                if len(exit_dates) == 0:
                    continue
                exit_date = exit_dates[0]

                if pd.isna(prices.loc[exit_date, split.symbol]):
                    continue
                exit_price = prices.loc[exit_date, split.symbol]

                ret = (exit_price - entry_price) / entry_price

                sector_roc = self.sector_roc_history.iloc[-1] if len(self.sector_roc_history) > 0 else 0

                samples.append([
                    split.split_factor,
                    sector_roc,
                    ret
                ])

            if len(samples) < 2:
                logger.warning(f"Insufficient training samples: {len(samples)}")
                return False

            samples = np.array(samples)
            X = samples[:, :2]  
            y = samples[:, 2]   

            self.model.fit(X, y)

            logger.info(f"Model trained on {len(samples)} samples")
            logger.info(f"Model coefficients: split_factor={self.model.coef_[0]:.4f}, "
                       f"sector_roc={self.model.coef_[1]:.4f}")
            logger.info(f"Model intercept: {self.model.intercept_:.4f}")

            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_return(self, split_factor: float, sector_roc: float) -> float:
        try:
            check_is_fitted(self.model)
            features = np.array([[split_factor, sector_roc]])
            return self.model.predict(features)[0]
        except NotFittedError:
            logger.warning("Model not fitted yet")
            return 0.0

    def calculate_order_quantity(self, symbol: str, exposure: float) -> int:
        try:
            account = self.get_account_info()
            portfolio_value = account['portfolio_value']

            price = self.get_current_price(symbol)
            if price is None or price <= 0:
                return 0

            target_value = portfolio_value * abs(exposure)

            shares = int(target_value / price)

            if exposure < 0:
                shares = -shares

            return shares

        except Exception as e:
            logger.error(f"Error calculating order quantity: {e}")
            return 0

    def place_order(self, symbol: str, quantity: int) -> Optional[str]:
        if quantity == 0:
            return None

        side = OrderSide.BUY if quantity > 0 else OrderSide.SELL

        if self.dry_run:
            logger.info(f"[DRY RUN] Would place order: {side.value} {abs(quantity)} {symbol}")
            return "dry_run_order_id"

        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=abs(quantity),
                side=side,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_request)
            logger.info(f"Order placed: {side.value} {abs(quantity)} {symbol} - ID: {order.id}")
            return str(order.id)

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None

    def get_open_positions(self) -> Dict[str, int]:
        try:
            positions = self.trading_client.get_all_positions()
            return {p.symbol: int(p.qty) for p in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def count_open_trades(self) -> int:
        return sum(len(trades) for trades in self.open_trades.values())

    def process_split_warning(self, split: SplitEvent) -> None:
        split_id = f"{split.symbol}_{split.execution_date.strftime('%Y%m%d')}"

        if split_id in self.processed_splits:
            return

        if self.count_open_trades() >= self.max_open_trades:
            logger.info(f"Max trades reached, skipping {split.symbol}")
            return

        try:
            check_is_fitted(self.model)
        except NotFittedError:
            logger.warning(f"Model not fitted, skipping {split.symbol}")
            return

        sector_roc = self.calculate_sector_roc()

        predicted_return = self.predict_return(split.split_factor, sector_roc)
        logger.info(f"Split warning: {split.symbol} factor={split.split_factor:.2f}, "
                   f"predicted_return={predicted_return:.4f}")

        if predicted_return == 0:
            return

        exposure = np.sign(predicted_return) * self.target_exposure_per_trade
        quantity = self.calculate_order_quantity(split.symbol, exposure)

        if quantity == 0:
            logger.info(f"Zero quantity calculated for {split.symbol}, skipping")
            return

        price = self.get_current_price(split.symbol)
        if price:
            order_value = abs(quantity) * price
            if order_value < self.min_order_value:
                logger.info(f"Order value ${order_value:.2f} below minimum "
                           f"${self.min_order_value}, skipping {split.symbol}")
                return

        order_id = self.place_order(split.symbol, quantity)

        if order_id:
            if split.symbol not in self.open_trades:
                self.open_trades[split.symbol] = []

            trade = Trade(
                symbol=split.symbol,
                quantity=quantity,
                entry_time=datetime.now(),
                close_time=datetime.now() + self.hold_duration,
                entry_price=price or 0
            )
            self.open_trades[split.symbol].append(trade)
            self.processed_splits.add(split_id)

            logger.info(f"Opened trade: {split.symbol} qty={quantity}, "
                       f"close_time={trade.close_time}")

    def check_for_occurred_splits(self) -> None:
        if not self.open_trades:
            return

        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        try:
            for split in self.polygon_client.list_splits(
                execution_date_gte=yesterday,
                execution_date_lte=today,
                limit=100
            ):
                symbol = split.ticker
                split_id = f"{symbol}_{split.execution_date}_occurred"

                if split_id in self.processed_occurred_splits:
                    continue

                if symbol in self.open_trades:
                    split_factor = split.split_to / split.split_from
                    for trade in self.open_trades[symbol]:
                        if not trade.closed:
                            old_qty = trade.quantity
                            trade.on_split_occurred(split_factor)
                            logger.info(f"Split occurred for {symbol}: adjusted qty {old_qty} -> {trade.quantity} "
                                       f"(factor={split_factor:.2f})")

                    self.processed_occurred_splits.add(split_id)

        except Exception as e:
            logger.error(f"Error checking for occurred splits: {e}")

    def scan_for_trade_exits(self) -> None:
        current_time = datetime.now()

        for symbol, trades in list(self.open_trades.items()):
            trades_to_remove = []

            for i, trade in enumerate(trades):
                if trade.should_close(current_time):
                    close_quantity = -trade.quantity  
                    order_id = self.place_order(symbol, close_quantity) 

                    if order_id:
                        trade.closed = True
                        trade.exit_price = self.get_current_price(symbol)
                        trades_to_remove.append(i)

                        if trade.entry_price and trade.exit_price:
                            pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                            pnl_pct = ((trade.exit_price - trade.entry_price) /
                                      trade.entry_price) * 100
                            logger.info(f"Closed trade: {symbol} P&L=${pnl:.2f} ({pnl_pct:.2f}%)")

            for i in reversed(trades_to_remove):
                del trades[i]

            if not trades:
                del self.open_trades[symbol]

    def run_daily_check(self) -> None:
        logger.info("=" * 50)
        logger.info("Running daily check...")

        try:
            account = self.get_account_info()
            logger.info(f"Account: Cash=${account['cash']:.2f}, "
                       f"Equity=${account['equity']:.2f}")
        except Exception as e:
            logger.error(f"Error getting account info: {e}")

        self.check_for_occurred_splits()

        self.scan_for_trade_exits()

        upcoming_splits = self.get_upcoming_splits(days_ahead=7)

        for split in upcoming_splits:
            self.process_split_warning(split)

        logger.info(f"Open trades: {self.count_open_trades()}/{self.max_open_trades}")
        logger.info("Daily check complete")

    def run_monthly_training(self) -> None:
        logger.info("Running monthly model retraining...")
        self.train_model()

    def start(self, run_once: bool = False) -> None:
        logger.info("Starting Split Trading Bot...")

        self.get_tech_universe()

        self.train_model()

        if run_once:
            self.run_daily_check()
            return

        schedule.every().monday.at("09:35").do(self.run_daily_check)
        schedule.every().tuesday.at("09:35").do(self.run_daily_check)
        schedule.every().wednesday.at("09:35").do(self.run_daily_check)
        schedule.every().thursday.at("09:35").do(self.run_daily_check)
        schedule.every().friday.at("09:35").do(self.run_daily_check)

        schedule.every().day.at("00:00").do(
            lambda: self.run_monthly_training() if datetime.now().day == 1 else None
        )

        schedule.every().monday.at("15:55").do(self.scan_for_trade_exits)
        schedule.every().tuesday.at("15:55").do(self.scan_for_trade_exits)
        schedule.every().wednesday.at("15:55").do(self.scan_for_trade_exits)
        schedule.every().thursday.at("15:55").do(self.scan_for_trade_exits)
        schedule.every().friday.at("15:55").do(self.scan_for_trade_exits)

        logger.info("Bot scheduled. Running...")

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)


def main():
    alpaca_api_key = os.environ.get('ALPACA_API_KEY')
    alpaca_secret_key = os.environ.get('ALPACA_SECRET_KEY')
    polygon_api_key = os.environ.get('POLYGON_API_KEY')

    if not all([alpaca_api_key, alpaca_secret_key, polygon_api_key]):
        print("Missing API keys. Please set environment variables:")
        print("  ALPACA_API_KEY")
        print("  ALPACA_SECRET_KEY")
        print("  POLYGON_API_KEY")
        print("\nOr create a .env file with these values.")
        return

    bot = SplitTradingBot(
        alpaca_api_key=alpaca_api_key,
        alpaca_secret_key=alpaca_secret_key,
        polygon_api_key=polygon_api_key,
        paper=True,           
        cash=1000.0,          
        max_open_trades=4,    
        hold_duration_days=3, 
        training_lookback_years=4,  
        min_order_value=50.0, 
        dry_run=False         
    )

    bot.start(run_once=False)


if __name__ == "__main__":
    main()
