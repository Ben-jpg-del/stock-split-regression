# CELL 1: IMPORTS AND INITIALIZATION

# QuantConnect imports
# from QuantConnect import *
# from QuantConnect.Research import *
# from AlgorithmImports import *

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# CELL 2: DATA STRUCTURES

@dataclass
class SplitEvent:
    symbol: str
    date: pd.Timestamp
    split_factor: float
    split_type: int = 0 


@dataclass
class TradeResult:
    symbol: str
    split_date: pd.Timestamp 
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    predicted_return: float
    actual_return: float
    direction: int


# CELL 3: BAR PERMUTATION

def get_permutation(ohlc, start_index=0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])
    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])
        perm_ohlc.append(perm_bars)

    return perm_ohlc if n_markets > 1 else perm_ohlc[0]


# CELL 4: STRATEGY IMPLEMENTATION

def calculate_sector_roc(prices: pd.Series, period: int = 22) -> pd.Series:
    return prices.pct_change(periods=period) * 100


def train_split_model(
    split_events: List[SplitEvent],
    prices_by_symbol: Dict[str, pd.DataFrame],
    sector_momentum: pd.Series,
    hold_duration: int,
    end_date: Optional[pd.Timestamp] = None
) -> Tuple[LinearRegression, int]:
    samples = []

    for split in split_events:
        if split.split_type == 1:   
            continue
        if end_date is not None and split.date >= end_date:
            continue
        if split.symbol not in prices_by_symbol:
            continue

        prices = prices_by_symbol[split.symbol]

        entry_mask = prices.index > split.date
        if not entry_mask.any():
            continue
        entry_series = prices.loc[entry_mask, 'open']
        if entry_series.empty or np.isnan(entry_series.iloc[0]):
            continue
        entry_price = entry_series.iloc[0]

        exit_date = split.date + pd.Timedelta(days=hold_duration)
        exit_mask = prices.index > exit_date
        if not exit_mask.any():
            continue
        exit_series = prices.loc[exit_mask, 'open']
        if exit_series.empty or np.isnan(exit_series.iloc[0]):
            continue
        exit_price = exit_series.iloc[0]

        momentum_mask = sector_momentum.index <= split.date
        if not momentum_mask.any():
            continue
        sector_roc = sector_momentum.loc[momentum_mask].iloc[-1]

        actual_return = (exit_price - entry_price) / entry_price
        samples.append([split.split_factor, sector_roc, actual_return])

    if len(samples) < 2:
        model = LinearRegression()
        model.coef_ = np.array([0, 0])
        model.intercept_ = 0
        return model, 0

    samples = np.array(samples)
    model = LinearRegression()
    model.fit(samples[:, :2], samples[:, -1])
    return model, len(samples)


def simulate_trades(
    model: LinearRegression,
    split_events: List[SplitEvent],
    prices_by_symbol: Dict[str, pd.DataFrame],
    sector_momentum: pd.Series,
    hold_duration: int,
    start_date: Optional[pd.Timestamp] = None,
    max_open_trades: int = 4,
    debug: bool = False
) -> List[TradeResult]:
    trades = []
    open_trades = []
    sorted_splits = sorted(split_events, key=lambda x: x.date)
    skipped_reasons = {'no_start': 0, 'no_symbol': 0, 'max_trades': 0,
                       'no_momentum': 0, 'zero_pred': 0, 'no_entry': 0, 'no_exit': 0,
                       'split_occurred': 0}

    for split in sorted_splits:
        if split.split_type == 1:  
            skipped_reasons['split_occurred'] += 1
            continue
        split_date = pd.Timestamp(split.date).tz_localize(None) if split.date.tzinfo else split.date

        if start_date is not None:
            start_dt = pd.Timestamp(start_date).tz_localize(None) if start_date.tzinfo else start_date
            if split_date < start_dt:
                skipped_reasons['no_start'] += 1
                continue

        if split.symbol not in prices_by_symbol:
            skipped_reasons['no_symbol'] += 1
            continue

        prices = prices_by_symbol[split.symbol].copy()
        if prices.index.tzinfo is not None:
            prices.index = prices.index.tz_localize(None)

        open_trades = [t for t in open_trades
                       if (t.split_date + pd.Timedelta(days=hold_duration)) > split_date]

        if len(open_trades) >= max_open_trades:
            skipped_reasons['max_trades'] += 1
            continue

        mom_index = sector_momentum.index
        if hasattr(mom_index, 'tz') and mom_index.tz is not None:
            mom_index = mom_index.tz_localize(None)
            sector_mom_clean = pd.Series(sector_momentum.values, index=mom_index)
        else:
            sector_mom_clean = sector_momentum

        momentum_mask = sector_mom_clean.index <= split_date
        if not momentum_mask.any():
            skipped_reasons['no_momentum'] += 1
            continue
        sector_roc = sector_mom_clean.loc[momentum_mask].iloc[-1]

        factors = np.array([[split.split_factor, sector_roc]])
        predicted_return = model.predict(factors)[0]

        if predicted_return == 0:
            skipped_reasons['zero_pred'] += 1
            continue

        entry_mask = prices.index > split_date
        if not entry_mask.any():
            skipped_reasons['no_entry'] += 1
            continue
        entry_series = prices.loc[entry_mask, 'open']
        if entry_series.empty or pd.isna(entry_series.iloc[0]):
            skipped_reasons['no_entry'] += 1
            continue
        entry_price = entry_series.iloc[0]
        entry_date = entry_series.index[0]

        exit_target_date = split_date + pd.Timedelta(days=hold_duration)
        exit_mask = prices.index > exit_target_date
        if not exit_mask.any():
            skipped_reasons['no_exit'] += 1
            continue
        exit_series = prices.loc[exit_mask, 'open']
        if exit_series.empty or pd.isna(exit_series.iloc[0]):
            skipped_reasons['no_exit'] += 1
            continue
        exit_price = exit_series.iloc[0]
        exit_date = exit_series.index[0]

        direction = 1 if predicted_return > 0 else -1
        actual_return = direction * (exit_price - entry_price) / entry_price

        trade = TradeResult(
            symbol=split.symbol,
            split_date=split_date,  
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            predicted_return=predicted_return,
            actual_return=actual_return,
            direction=direction
        )
        trades.append(trade)
        open_trades.append(trade)

    if debug:
        print(f"  Trade simulation: {len(trades)} trades generated")
        print(f"  Skipped reasons: {skipped_reasons}")

    return trades


def calculate_profit_factor(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0  
    pf = gains / losses
    return min(pf, 10.0)  


def evaluate_split_strategy(
    prices_by_symbol: Dict[str, pd.DataFrame],
    split_events: List[SplitEvent],
    sector_momentum: pd.Series,
    hold_duration: int = 3,
    train_end_date: Optional[pd.Timestamp] = None,
    max_open_trades: int = 4,
    debug: bool = False
) -> float:
    if not split_events:
        if debug:
            print("DEBUG: No split events")
        return 0.0

    sorted_splits = sorted(split_events, key=lambda x: x.date)

    if train_end_date is None:
        n_train = int(len(sorted_splits) * 0.7)
        train_end_date = sorted_splits[n_train].date

    train_splits = [s for s in sorted_splits if s.date < train_end_date]
    test_splits = [s for s in sorted_splits if s.date >= train_end_date]

    if debug:
        print(f"DEBUG: Train end date: {train_end_date}")
        print(f"DEBUG: Training on {len(train_splits)} splits, testing on {len(test_splits)} splits")
        train_warnings = sum(1 for s in train_splits if s.split_type == 0)
        test_warnings = sum(1 for s in test_splits if s.split_type == 0)
        print(f"DEBUG: Training WARNING splits: {train_warnings}, Testing WARNING splits: {test_warnings}")

    if len(train_splits) < 2:
        if debug:
            print(f"DEBUG: Not enough training splits: {len(train_splits)}")
        return 0.0

    model, n_samples = train_split_model(
        split_events=train_splits,
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=hold_duration,
        end_date=train_end_date
    )

    if debug:
        print(f"DEBUG: Model trained on {n_samples} valid samples")
        if n_samples >= 2:
            print(f"DEBUG: Model coefficients: split_factor={model.coef_[0]:.4f}, sector_roc={model.coef_[1]:.4f}")

    if n_samples < 2:
        if debug:
            print("DEBUG: Not enough training samples")
        return 0.0

    trades = simulate_trades(
        model=model,
        split_events=test_splits,
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=hold_duration,
        start_date=train_end_date,
        max_open_trades=max_open_trades,
        debug=debug
    )

    if debug:
        print(f"DEBUG: Generated {len(trades)} trades")
        if trades:
            returns = pd.Series([t.actual_return for t in trades])
            print(f"DEBUG: Trade returns - mean: {returns.mean():.4f}, win_rate: {(returns > 0).mean():.2%}")

    if not trades:
        return 0.0

    returns = pd.Series([t.actual_return for t in trades])
    return calculate_profit_factor(returns)


# CELL 5: MCPT RUNNER

def run_split_events_mcpt(
    prices_by_symbol: Dict[str, pd.DataFrame],
    split_events: List[SplitEvent],
    sector_momentum: pd.Series,
    hold_duration: int = 3,
    train_end_date: Optional[pd.Timestamp] = None,
    n_permutations: int = 500,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)

    warning_splits = [s for s in split_events if s.split_type == 0]
    occurred_splits = [s for s in split_events if s.split_type == 1]

    print(f"Total split events: {len(split_events)}")
    print(f"  - WARNING (tradeable): {len(warning_splits)}")
    print(f"  - SPLIT_OCCURRED (ignored): {len(occurred_splits)}")
    print(f"Total symbols with prices: {len(prices_by_symbol)}")

    if train_end_date:
        train_warnings = sum(1 for s in warning_splits if s.date < train_end_date)
        test_warnings = sum(1 for s in warning_splits if s.date >= train_end_date)
        print(f"Train end date: {train_end_date.date()}")
        print(f"  - Training WARNING splits: {train_warnings}")
        print(f"  - Testing WARNING splits: {test_warnings} (potential trades)")

    print("\nEvaluating strategy on real data (with debug)...")
    real_metric = evaluate_split_strategy(
        prices_by_symbol=prices_by_symbol,
        split_events=split_events,
        sector_momentum=sector_momentum,
        hold_duration=hold_duration,
        train_end_date=train_end_date,
        debug=True  # Enable debug for first run
    )
    print(f"\nReal Profit Factor: {real_metric:.4f}")

    perm_better_count = 1
    permuted_metrics = []

    print(f"\nRunning {n_permutations} permutations...")
    for i in range(1, n_permutations):
        if i % max(1, n_permutations // 10) == 0:
            print(f"  Progress: {i}/{n_permutations} ({100*i//n_permutations}%)")

        permuted_prices = {
            symbol: get_permutation(ohlc)
            for symbol, ohlc in prices_by_symbol.items()
        }

        perm_metric = evaluate_split_strategy(
            prices_by_symbol=permuted_prices,
            split_events=split_events,
            sector_momentum=sector_momentum,
            hold_duration=hold_duration,
            train_end_date=train_end_date
        )

        if perm_metric >= real_metric:
            perm_better_count += 1
        permuted_metrics.append(perm_metric)

    print(f"  Progress: {n_permutations}/{n_permutations} (100%)")
    p_value = perm_better_count / n_permutations

    return {
        'real_metric': real_metric,
        'permuted_metrics': permuted_metrics,
        'p_value': p_value,
        'n_permutations': n_permutations
    }


# CELL 6: VISUALIZATION FUNCTIONS

def plot_mcpt_results(results: Dict[str, Any], figsize=(14, 5)):
    permuted = np.array(results['permuted_metrics'])
    permuted = permuted[np.isfinite(permuted)]

    if len(permuted) == 0:
        print("No valid permutation results to plot")
        return

    real_metric = results['real_metric']
    if not np.isfinite(real_metric):
        real_metric = 10.0  

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    pd.Series(permuted).hist(
        ax=ax, bins=30, color='steelblue', alpha=0.7, edgecolor='black'
    )
    ax.axvline(real_metric, color='red', linewidth=2,
               label=f'Real: {real_metric:.4f}')
    ax.axvline(np.mean(permuted), color='green',
               linewidth=2, linestyle='--',
               label=f'Mean: {np.mean(permuted):.4f}')
    ax.set_xlabel('Profit Factor')
    ax.set_ylabel('Frequency')
    ax.set_title(f'MCPT Results | P-Value: {results["p_value"]:.4f}')
    ax.legend()

    ax = axes[1]
    sorted_metrics = np.sort(permuted)
    cumulative = np.arange(1, len(sorted_metrics) + 1) / len(sorted_metrics)
    ax.plot(sorted_metrics, cumulative, 'b-', linewidth=2)
    ax.axvline(real_metric, color='red', linewidth=2)
    percentile = (1 - results['p_value']) * 100
    ax.axhline(1 - results['p_value'], color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Profit Factor')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'CDF (Percentile: {percentile:.1f}%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if results['p_value'] < 0.01:
        print(f"P-Value: {results['p_value']:.4f} - HIGHLY SIGNIFICANT (p < 0.01)")
        print("Strong evidence that strategy performance is NOT due to chance.")
    elif results['p_value'] < 0.05:
        print(f"P-Value: {results['p_value']:.4f} - SIGNIFICANT (p < 0.05)")
        print("Strategy performance is statistically significant.")
    elif results['p_value'] < 0.10:
        print(f"P-Value: {results['p_value']:.4f} - MARGINALLY SIGNIFICANT (p < 0.10)")
        print("Weak evidence of genuine strategy edge.")
    else:
        print(f"P-Value: {results['p_value']:.4f} - NOT SIGNIFICANT (p >= 0.10)")
        print("Performance could be explained by random chance.")


def plot_trade_analysis(trades: List[TradeResult], figsize=(14, 10)):
    if not trades:
        print("No trades to analyze")
        return

    returns = pd.Series([t.actual_return for t in trades])
    predictions = pd.Series([t.predicted_return for t in trades])

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    returns.hist(ax=ax, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.axvline(returns.mean(), color='green', label=f'Mean: {returns.mean()*100:.2f}%')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution')
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(predictions, returns, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.axvline(0, color='red', linestyle='--')
    corr = predictions.corr(returns)
    ax.set_xlabel('Predicted Return')
    ax.set_ylabel('Actual Return')
    ax.set_title(f'Predicted vs Actual (Corr: {corr:.3f})')

    ax = axes[1, 0]
    cum_returns = (1 + returns).cumprod()
    cum_returns.plot(ax=ax)
    ax.axhline(1, color='red', linestyle='--')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Equity Curve')

    ax = axes[1, 1]
    ax.axis('off')
    pf = calculate_profit_factor(returns)
    pf_str = f"{pf:.2f}" if np.isfinite(pf) else "N/A"
    stats_text = f"""
    STRATEGY STATISTICS
    {'='*40}

    Total Trades:     {len(trades)}
    Win Rate:         {(returns > 0).mean()*100:.1f}%

    Mean Return:      {returns.mean()*100:.2f}%
    Std Return:       {returns.std()*100:.2f}%

    Profit Factor:    {pf_str}
    Total Return:     {((1+returns).prod()-1)*100:.1f}%

    Best Trade:       {returns.max()*100:.2f}%
    Worst Trade:      {returns.min()*100:.2f}%

    Long Trades:      {sum(1 for t in trades if t.direction == 1)}
    Short Trades:     {sum(1 for t in trades if t.direction == -1)}
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=11, verticalalignment='center')

    plt.suptitle('Split Events Strategy - Trade Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()


# CELL 7: QUANTCONNECT DATA FETCHING

def fetch_data_from_quantconnect(qb, start_date, end_date, lookback_years=4):
    print("Fetching data from QuantConnect...")

    xlk = qb.add_equity("XLK").symbol

    sector_history = qb.history(
        xlk,
        start_date - timedelta(days=lookback_years*365 + 30),
        end_date,
        Resolution.DAILY
    )

    if isinstance(sector_history.index, pd.MultiIndex):
        sector_close = sector_history['close'].droplevel(0)
    else:
        sector_close = sector_history['close']

    sector_momentum = calculate_sector_roc(sector_close, period=22)
    print(f"Sector momentum calculated: {len(sector_momentum)} days")

    tech_symbols = []
    symbol_map = {}  

    print("Fetching full tech sector universe...")

    try:
        def tech_sector_filter(fundamental):
            return [
                x.symbol for x in fundamental
                if (x.asset_classification.morningstar_sector_code == MorningstarSectorCode.TECHNOLOGY
                    and x.has_fundamental_data
                    and x.price > 1)  
            ]

        universe_history = qb.universe_history(
            Fundamentals,
            start_date,
            end_date,
            tech_sector_filter
        )

        if universe_history is not None:
            all_tech_symbols = set()
            for date_symbols in universe_history:
                for symbol in date_symbols:
                    ticker = str(symbol).split()[0] if ' ' in str(symbol) else str(symbol.value)
                    all_tech_symbols.add(ticker)

            print(f"Found {len(all_tech_symbols)} tech stocks from universe history")

            for ticker in all_tech_symbols:
                try:
                    equity = qb.add_equity(ticker)
                    tech_symbols.append(equity.symbol)
                    symbol_map[ticker] = equity.symbol
                except:
                    pass

    except Exception as e:
        print(f"Could not use universe_history: {e}")
        print("Trying alternative method...")

        try:
            coarse_data = qb.universe_history(
                CoarseFundamental,
                start_date,
                end_date
            )

            candidate_symbols = set()
            if coarse_data is not None:
                for date_data in coarse_data:
                    for cf in sorted(date_data, key=lambda x: x.dollar_volume, reverse=True)[:2000]:
                        if cf.has_fundamental_data and cf.price > 1:
                            candidate_symbols.add(cf.symbol)

            print(f"Found {len(candidate_symbols)} candidates from coarse universe")

            if candidate_symbols:
                fine_data = qb.universe_history(
                    FineFundamental,
                    start_date,
                    end_date,
                    list(candidate_symbols)
                )

                if fine_data is not None:
                    for date_data in fine_data:
                        for ff in date_data:
                            if ff.asset_classification.morningstar_sector_code == MorningstarSectorCode.TECHNOLOGY:
                                ticker = str(ff.symbol).split()[0]
                                if ticker not in symbol_map:
                                    try:
                                        equity = qb.add_equity(ticker)
                                        tech_symbols.append(equity.symbol)
                                        symbol_map[ticker] = equity.symbol
                                    except:
                                        pass

        except Exception as e2:
            print(f"Could not use coarse/fine universe: {e2}")

    fallback_tech_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
        'AVGO', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'ACN', 'INTC', 'AMD', 'QCOM',
        'TXN', 'IBM', 'INTU', 'NOW', 'AMAT', 'LRCX', 'MU', 'ADI', 'KLAC',
        'MRVL', 'ANSS', 'ADSK', 'WDAY', 'CTSH', 'FISV', 'PAYX', 'CDNS',
        'SNPS', 'NXPI', 'MCHP', 'ON', 'MPWR', 'SWKS', 'QRVO', 'TER',
        'FTNT', 'PANW', 'CRWD', 'ZS', 'DDOG', 'SNOW', 'NET', 'MDB',
        'TEAM', 'SPLK', 'OKTA', 'DOCU', 'ZM', 'ROKU', 'SQ', 'SHOP',
        'PLTR', 'PATH', 'U', 'NFLX', 'PYPL', 'UBER', 'ABNB', 'COIN',
        'HOOD', 'RBLX', 'SNAP', 'PINS', 'TWLO', 'ESTC', 'HUBS', 'TTD',
        'VEEV', 'COUP', 'FIVN', 'BILL', 'GTLB', 'MNDY', 'CFLT', 'DOMO',
        'APPS', 'UPST', 'AFRM', 'SOFI', 'LCID', 'RIVN', 'NIO', 'XPEV',
        'TSM', 'ASML', 'LRCX', 'AMAT', 'KLAC', 'ENTG', 'MKSI', 'CRUS',
        'SLAB', 'DIOD', 'AOSL', 'POWI', 'SMTC', 'RMBS', 'SITM', 'FORM',
        'MSFT', 'ORCL', 'SAP', 'NOW', 'INTU', 'ADBE', 'CRM', 'WDAY',
        'TEAM', 'ZEN', 'HUBS', 'DOCU', 'BILL', 'MNDY', 'FROG', 'SUMO',
        'AAPL', 'DELL', 'HPQ', 'HPE', 'NTAP', 'PSTG', 'WDC', 'STX',
        'LOGI', 'CRSR', 'HEAR', 'GPRO', 'SONO', 'IRBT', 'ARLO',
    ]

    added_from_fallback = 0
    for ticker in fallback_tech_tickers:
        if ticker not in symbol_map:
            try:
                equity = qb.add_equity(ticker)
                tech_symbols.append(equity.symbol)
                symbol_map[ticker] = equity.symbol
                added_from_fallback += 1
            except:
                pass

    print(f"Added {len(tech_symbols)} total tech stocks ({added_from_fallback} from fallback list)")

    split_events = []
    lookback = timedelta(days=lookback_years * 365)

    # Calculate the full date range needed for splits
    # Training period: (start_date - lookback) to start_date
    # Testing period: start_date to end_date
    splits_start = start_date - lookback
    splits_end = end_date

    print(f"Fetching splits from {splits_start.date()} to {splits_end.date()}...")

    try:
        # Use explicit date range instead of relative lookback
        splits_df = qb.history(Split, tech_symbols, splits_start, splits_end)

        if splits_df is not None and not splits_df.empty:
            if isinstance(splits_df.index, pd.MultiIndex):
                splits_df = splits_df.reset_index()

            for idx, row in splits_df.iterrows():
                try:
                    if 'symbol' in splits_df.columns:
                        sym = str(row['symbol'])
                    elif 'Symbol' in splits_df.columns:
                        sym = str(row['Symbol'])
                    else:
                        sym = str(row.iloc[0]) if len(row) > 0 else None

                    if sym is None:
                        continue

                    if 'splitfactor' in splits_df.columns:
                        factor = row['splitfactor']
                    elif 'SplitFactor' in splits_df.columns:
                        factor = row['SplitFactor']
                    elif 'value' in splits_df.columns:
                        factor = row['value']
                    else:
                        factor = 0.5  

                    if 'time' in splits_df.columns:
                        split_date = pd.Timestamp(row['time'])
                    elif 'Time' in splits_df.columns:
                        split_date = pd.Timestamp(row['Time'])
                    elif 'endtime' in splits_df.columns:
                        split_date = pd.Timestamp(row['endtime'])
                    else:
                        split_date = pd.Timestamp(splits_df.index[idx])

                    split_type = 0  
                    if 'type' in splits_df.columns:
                        split_type = int(row['type'])
                    elif 'Type' in splits_df.columns:
                        split_type = int(row['Type'])
                    elif 'splittype' in splits_df.columns:
                        split_type = int(row['splittype'])

                    split_events.append(SplitEvent(
                        symbol=sym,
                        date=split_date,
                        split_factor=float(factor),
                        split_type=split_type
                    ))
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Could not fetch splits via history: {e}")
        print("Generating synthetic split events for testing...")

        import random
        random.seed(42)
        # Calculate total days from splits_start to splits_end
        total_days = (splits_end - splits_start).days
        for ticker, symbol in symbol_map.items():
            n_splits = random.randint(2, 5)
            for _ in range(n_splits):
                # Generate splits across the full date range (training + testing)
                days_from_start = random.randint(100, total_days - 100)
                split_date = splits_start + timedelta(days=days_from_start)
                split_events.append(SplitEvent(
                    symbol=ticker,
                    date=pd.Timestamp(split_date),
                    split_factor=random.choice([0.5, 0.25, 0.2, 2.0, 4.0]),
                    split_type=0
                ))

    print(f"Found {len(split_events)} split events")

    warning_count = sum(1 for s in split_events if s.split_type == 0)
    occurred_count = sum(1 for s in split_events if s.split_type == 1)
    print(f"  - WARNING (type 0): {warning_count} (used for trading)")
    print(f"  - SPLIT_OCCURRED (type 1): {occurred_count} (ignored)")

    if split_events:
        print(f"Sample splits: {[(s.symbol, s.date.date(), s.split_factor, 'WARN' if s.split_type==0 else 'OCCURRED') for s in split_events[:3]]}")

    prices_by_symbol = {}

    for ticker, symbol in symbol_map.items():
        try:
            history = qb.history(
                symbol,
                start_date - timedelta(days=lookback_years*365),
                end_date,
                Resolution.DAILY
            )

            if history is not None and not history.empty:
                if isinstance(history.index, pd.MultiIndex):
                    price_df = history[['open', 'high', 'low', 'close']].droplevel(0)
                else:
                    price_df = history[['open', 'high', 'low', 'close']]

                prices_by_symbol[ticker] = price_df
        except Exception as e:
            print(f"Could not fetch prices for {ticker}: {e}")

    print(f"Fetched prices for {len(prices_by_symbol)} symbols")

    updated_splits = []
    for split in split_events:
        sym = split.symbol
        if ' ' in sym:
            sym = sym.split(' ')[0]
        if sym in prices_by_symbol:
            updated_splits.append(SplitEvent(
                symbol=sym,
                date=split.date,
                split_factor=split.split_factor,
                split_type=split.split_type
            ))

    if updated_splits:
        split_events = updated_splits
        print(f"Matched {len(split_events)} splits to price data")

    return prices_by_symbol, split_events, sector_momentum


# CELL 8: MAIN EXECUTION

QUANTBOOK_MAIN_CODE = '''
qb = QuantBook()

START_DATE = datetime(2023, 1, 1)  
END_DATE = datetime(2025, 12, 24)  
LOOKBACK_YEARS = 4  
HOLD_DURATION = 3  
N_PERMUTATIONS = 200  

prices_by_symbol, split_events, sector_momentum = fetch_data_from_quantconnect(
    qb, START_DATE, END_DATE, LOOKBACK_YEARS
)

results = run_split_events_mcpt(
    prices_by_symbol=prices_by_symbol,
    split_events=split_events,
    sector_momentum=sector_momentum,
    hold_duration=HOLD_DURATION,
    train_end_date=pd.Timestamp(START_DATE),  
    n_permutations=N_PERMUTATIONS,
    seed=42
)

plot_mcpt_results(results)

train_end = pd.Timestamp(START_DATE)
train_splits = [s for s in split_events if s.date < train_end]
test_splits = [s for s in split_events if s.date >= train_end]

model, n_samples = train_split_model(
    split_events=train_splits,
    prices_by_symbol=prices_by_symbol,
    sector_momentum=sector_momentum,
    hold_duration=HOLD_DURATION,
    end_date=train_end
)

trades = simulate_trades(
    model=model,
    split_events=test_splits,
    prices_by_symbol=prices_by_symbol,
    sector_momentum=sector_momentum,
    hold_duration=HOLD_DURATION,
    start_date=train_end,
    debug=True  # Show skip reasons
)

plot_trade_analysis(trades)

# Print model coefficients and trade summary
print("\\nModel Coefficients:")
print(f"  Split Factor:    {model.coef_[0]:.6f}")
print(f"  Sector Momentum: {model.coef_[1]:.6f}")
print(f"  Intercept:       {model.intercept_:.6f}")
print(f"\\nTrade Summary:")
print(f"  Training samples: {n_samples}")
print(f"  Total trades: {len(trades)}")
print(f"  Total orders: {len(trades) * 2} (entry + exit)")
'''

# CELL 9: LOCAL TESTING

def run_local_demo():
    print("="*60)
    print("Split Events Strategy - MCPT Demo (Synthetic Data)")
    print("="*60)

    np.random.seed(42)

    n_days = 1000
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMD']

    prices_by_symbol = {}
    for symbol in symbols:
        close = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
        high = close * (1 + np.abs(np.random.randn(n_days) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n_days) * 0.01))
        open_ = np.roll(close, 1) * (1 + np.random.randn(n_days) * 0.005)
        open_[0] = close[0]

        prices_by_symbol[symbol] = pd.DataFrame({
            'open': open_, 'high': high, 'low': low, 'close': close
        }, index=dates)

    sector_prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.015))
    sector_momentum = calculate_sector_roc(pd.Series(sector_prices, index=dates), period=22)

    split_events = []
    for symbol in symbols:
        n_splits = np.random.randint(5, 11)
        split_dates = np.random.choice(dates[100:-100], n_splits, replace=False)
        for split_date in sorted(split_dates):
            split_events.append(SplitEvent(
                symbol=symbol,
                date=pd.Timestamp(split_date),
                split_factor=np.random.choice([0.5, 0.25, 0.2, 0.1, 2.0, 4.0]),
                split_type=0  
            ))

    print(f"\nGenerated {len(split_events)} splits across {len(symbols)} symbols")

    train_end_date = dates[int(len(dates) * 0.7)]

    results = run_split_events_mcpt(
        prices_by_symbol=prices_by_symbol,
        split_events=split_events,
        sector_momentum=sector_momentum,
        hold_duration=3,
        train_end_date=pd.Timestamp(train_end_date),
        n_permutations=100,
        seed=42
    )

    plot_mcpt_results(results)

    train_splits = [s for s in split_events if s.date < pd.Timestamp(train_end_date)]
    test_splits = [s for s in split_events if s.date >= pd.Timestamp(train_end_date)]

    model, _ = train_split_model(
        split_events=train_splits,
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=3,
        end_date=pd.Timestamp(train_end_date)
    )

    trades = simulate_trades(
        model=model,
        split_events=test_splits,
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=3,
        start_date=pd.Timestamp(train_end_date),
        debug=True
    )

    if trades:
        plot_trade_analysis(trades)
        print(f"\nTotal trades: {len(trades)}, Total orders: {len(trades) * 2}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TO RUN ON QUANTCONNECT:")
    print("="*60)
    print(QUANTBOOK_MAIN_CODE)
    print("\n" + "="*60)
    print("RUNNING LOCAL DEMO...")
    print("="*60 + "\n")

    run_local_demo()
