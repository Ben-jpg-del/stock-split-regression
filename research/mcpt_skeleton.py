import numpy as np
import pandas as pd
from typing import Callable, Optional, List, Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

from bar_permute import get_permutation


@dataclass
class SplitEvent:
    symbol: str
    date: pd.Timestamp
    split_factor: float
    warning_date: Optional[pd.Timestamp] = None


@dataclass
class TradeResult:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    predicted_return: float
    actual_return: float
    direction: int  # 1 for long, -1 for short


def run_insample_mcpt(
    ohlc_data: pd.DataFrame,
    strategy_metric_fn: Callable[[pd.DataFrame], float],
    n_permutations: int = 1000,
    seed: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)

    real_metric = strategy_metric_fn(ohlc_data)

    perm_better_count = 1  # Start at 1 (include real in count)
    permuted_metrics = []

    iterator = range(1, n_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc="In-Sample MCPT")

    for _ in iterator:
        permuted_data = get_permutation(ohlc_data)
        perm_metric = strategy_metric_fn(permuted_data)

        if perm_metric >= real_metric:
            perm_better_count += 1

        permuted_metrics.append(perm_metric)

    p_value = perm_better_count / n_permutations

    return {
        'real_metric': real_metric,
        'permuted_metrics': permuted_metrics,
        'p_value': p_value,
        'n_permutations': n_permutations
    }


def run_walkforward_mcpt(
    ohlc_data: pd.DataFrame,
    strategy_metric_fn: Callable[[pd.DataFrame, int], float],
    train_window: int,
    n_permutations: int = 200,
    seed: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)

    real_metric = strategy_metric_fn(ohlc_data, train_window)

    perm_better_count = 1
    permuted_metrics = []

    iterator = range(1, n_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc="Walk-Forward MCPT")

    for _ in iterator:
        permuted_data = get_permutation(ohlc_data, start_index=train_window)
        perm_metric = strategy_metric_fn(permuted_data, train_window)

        if perm_metric >= real_metric:
            perm_better_count += 1

        permuted_metrics.append(perm_metric)

    p_value = perm_better_count / n_permutations

    return {
        'real_metric': real_metric,
        'permuted_metrics': permuted_metrics,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'train_window': train_window
    }


def run_split_events_mcpt(
    prices_by_symbol: Dict[str, pd.DataFrame],
    split_events: List[SplitEvent],
    sector_momentum: pd.Series,
    hold_duration: int = 3,
    train_test_split: float = 0.7,
    n_permutations: int = 500,
    seed: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)

    real_metric = evaluate_split_events_strategy(
        prices_by_symbol=prices_by_symbol,
        split_events=split_events,
        sector_momentum=sector_momentum,
        hold_duration=hold_duration,
        train_test_split=train_test_split
    )

    perm_better_count = 1
    permuted_metrics = []

    iterator = range(1, n_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc="Split Events MCPT")

    for _ in iterator:
        permuted_prices = {}
        for symbol, ohlc in prices_by_symbol.items():
            permuted_prices[symbol] = get_permutation(ohlc)

        perm_metric = evaluate_split_events_strategy(
            prices_by_symbol=permuted_prices,
            split_events=split_events,
            sector_momentum=sector_momentum,
            hold_duration=hold_duration,
            train_test_split=train_test_split
        )

        if perm_metric >= real_metric:
            perm_better_count += 1

        permuted_metrics.append(perm_metric)

    p_value = perm_better_count / n_permutations

    return {
        'real_metric': real_metric,
        'permuted_metrics': permuted_metrics,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'hold_duration': hold_duration,
        'train_test_split': train_test_split
    }


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
        entry_date = entry_series.index[0]

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

        samples.append([
            split.split_factor,
            sector_roc,
            actual_return
        ])

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
    max_open_trades: int = 4
) -> List[TradeResult]:
    trades = []
    open_trades = []

    sorted_splits = sorted(split_events, key=lambda x: x.date)

    for split in sorted_splits:
        if start_date is not None and split.date < start_date:
            continue

        if split.symbol not in prices_by_symbol:
            continue

        prices = prices_by_symbol[split.symbol]

        close_date = split.date
        open_trades = [t for t in open_trades if t.exit_date > close_date]

        if len(open_trades) >= max_open_trades:
            continue

        momentum_mask = sector_momentum.index <= split.date
        if not momentum_mask.any():
            continue
        sector_roc = sector_momentum.loc[momentum_mask].iloc[-1]

        factors = np.array([[split.split_factor, sector_roc]])
        predicted_return = model.predict(factors)[0]

        if predicted_return == 0:
            continue

        entry_mask = prices.index > split.date
        if not entry_mask.any():
            continue
        entry_series = prices.loc[entry_mask, 'open']
        if entry_series.empty or np.isnan(entry_series.iloc[0]):
            continue
        entry_price = entry_series.iloc[0]
        entry_date = entry_series.index[0]

        exit_target_date = split.date + pd.Timedelta(days=hold_duration)
        exit_mask = prices.index > exit_target_date
        if not exit_mask.any():
            continue
        exit_series = prices.loc[exit_mask, 'open']
        if exit_series.empty or np.isnan(exit_series.iloc[0]):
            continue
        exit_price = exit_series.iloc[0]
        exit_date = exit_series.index[0]

        direction = 1 if predicted_return > 0 else -1
        actual_return = direction * (exit_price - entry_price) / entry_price

        trade = TradeResult(
            symbol=split.symbol,
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

    return trades


def evaluate_split_events_strategy(
    prices_by_symbol: Dict[str, pd.DataFrame],
    split_events: List[SplitEvent],
    sector_momentum: pd.Series,
    hold_duration: int = 3,
    train_test_split: float = 0.7,
    max_open_trades: int = 4,
    metric: str = 'profit_factor'
) -> float:
    if not split_events:
        return 0.0

    sorted_splits = sorted(split_events, key=lambda x: x.date)

    n_train = int(len(sorted_splits) * train_test_split)
    if n_train < 2:
        return 0.0

    train_end_date = sorted_splits[n_train].date

    model, n_samples = train_split_model(
        split_events=sorted_splits[:n_train],
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=hold_duration,
        end_date=train_end_date
    )

    if n_samples < 2:
        return 0.0

    trades = simulate_trades(
        model=model,
        split_events=sorted_splits[n_train:],
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=hold_duration,
        start_date=train_end_date,
        max_open_trades=max_open_trades
    )

    if not trades:
        return 0.0

    returns = pd.Series([t.actual_return for t in trades])

    if metric == 'profit_factor':
        return calculate_profit_factor(returns)
    elif metric == 'sharpe':
        return calculate_sharpe_ratio(returns, periods_per_year=52)  # Assume weekly trades
    elif metric == 'total_return':
        return (1 + returns).prod() - 1
    elif metric == 'win_rate':
        return (returns > 0).mean()
    else:
        return calculate_profit_factor(returns)


def calculate_profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:   
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def calculate_sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0].std()
    if downside == 0 or np.isnan(downside):
        return float('inf') if returns.mean() > 0 else 0.0
    return returns.mean() / downside * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return drawdowns.min()


def plot_mcpt_results(
    results: Dict[str, Any],
    metric_name: str = "Profit Factor",
    title: Optional[str] = None,
    style: str = 'dark_background'
) -> None:
    plt.style.use(style)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pd.Series(results['permuted_metrics']).hist(
        ax=axes[0], color='steelblue', alpha=0.7, label='Permutations', bins=30, edgecolor='black'
    )
    axes[0].axvline(
        results['real_metric'], color='red', linewidth=2,
        label=f'Real ({results["real_metric"]:.4f})'
    )
    axes[0].axvline(
        np.mean(results['permuted_metrics']), color='green', linewidth=2, linestyle='--',
        label=f'Perm Mean ({np.mean(results["permuted_metrics"]):.4f})'
    )
    axes[0].set_xlabel(metric_name)
    axes[0].set_ylabel("Frequency")
    if title is None:
        title = f"MCPT Results | P-Value: {results['p_value']:.4f}"
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(False)

    sorted_metrics = np.sort(results['permuted_metrics'])
    cumulative = np.arange(1, len(sorted_metrics) + 1) / len(sorted_metrics)
    axes[1].plot(sorted_metrics, cumulative, 'b-', linewidth=2)
    axes[1].axvline(results['real_metric'], color='red', linewidth=2, label='Real')
    percentile = (1 - results['p_value']) * 100
    axes[1].axhline(1 - results['p_value'], color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel(metric_name)
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title(f"CDF (Percentile: {percentile:.1f}%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_trade_analysis(trades: List[TradeResult], title: str = "Trade Analysis") -> None:
    if not trades:
        print("No trades to analyze")
        return

    returns = pd.Series([t.actual_return for t in trades])
    predictions = pd.Series([t.predicted_return for t in trades])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    returns.hist(ax=axes[0, 0], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--')
    axes[0, 0].axvline(returns.mean(), color='green', linestyle='-',
                       label=f'Mean: {returns.mean()*100:.2f}%')
    axes[0, 0].set_xlabel('Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Return Distribution')
    axes[0, 0].legend()

    axes[0, 1].scatter(predictions, returns, alpha=0.5)
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Return')
    axes[0, 1].set_ylabel('Actual Return')
    axes[0, 1].set_title(f'Predicted vs Actual (Corr: {predictions.corr(returns):.3f})')

    cum_returns = (1 + returns).cumprod()
    cum_returns.plot(ax=axes[1, 0])
    axes[1, 0].axhline(1, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Trade Number')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].set_title('Equity Curve')

    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    labels = ['Long', 'Short']
    win_rates = [
        sum(1 for t in long_trades if t.actual_return > 0) / len(long_trades) if long_trades else 0,
        sum(1 for t in short_trades if t.actual_return > 0) / len(short_trades) if short_trades else 0
    ]
    counts = [len(long_trades), len(short_trades)]

    x = np.arange(len(labels))
    width = 0.35
    axes[1, 1].bar(x - width/2, win_rates, width, label='Win Rate', color='green', alpha=0.7)
    axes[1, 1].bar(x + width/2, [c/max(counts) for c in counts], width, label='Trade Count (scaled)', color='blue', alpha=0.7)
    axes[1, 1].axhline(0.5, color='red', linestyle='--')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].set_title(f'Performance by Direction (n={len(trades)})')
    axes[1, 1].legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Split Events Strategy - MCPT Demonstration")
    print("=" * 60)

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
            'open': open_,
            'high': high,
            'low': low,
            'close': close
        }, index=dates)

    sector_prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.015))
    sector_momentum = calculate_sector_roc(pd.Series(sector_prices, index=dates), period=22)

    split_events = []
    for i, symbol in enumerate(symbols):
        n_splits = np.random.randint(5, 11)
        split_dates = np.random.choice(dates[100:-100], n_splits, replace=False)
        for split_date in sorted(split_dates):
            split_events.append(SplitEvent(
                symbol=symbol,
                date=pd.Timestamp(split_date),
                split_factor=np.random.choice([0.5, 0.25, 0.2, 0.1, 2.0, 4.0, 5.0])
            ))

    print(f"\nGenerated {len(split_events)} synthetic split events across {len(symbols)} symbols")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    print("\nEvaluating strategy on real data...")
    real_metric = evaluate_split_events_strategy(
        prices_by_symbol=prices_by_symbol,
        split_events=split_events,
        sector_momentum=sector_momentum,
        hold_duration=3,
        train_test_split=0.7
    )
    print(f"Real Profit Factor: {real_metric:.4f}")

    print(f"\nRunning MCPT with 100 permutations (demo)...")
    results = run_split_events_mcpt(
        prices_by_symbol=prices_by_symbol,
        split_events=split_events,
        sector_momentum=sector_momentum,
        hold_duration=3,
        train_test_split=0.7,
        n_permutations=100,
        seed=42,
        show_progress=True
    )

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Real Profit Factor: {results['real_metric']:.4f}")
    print(f"Permutation Mean:   {np.mean(results['permuted_metrics']):.4f}")
    print(f"Permutation Std:    {np.std(results['permuted_metrics']):.4f}")
    print(f"P-Value:            {results['p_value']:.4f}")
    print(f"\nInterpretation: ", end="")

    if results['p_value'] < 0.05:
        print("Strategy is STATISTICALLY SIGNIFICANT (p < 0.05)")
        print("The model's predictions are unlikely due to chance.")
    elif results['p_value'] < 0.10:
        print("Strategy shows MARGINAL significance (0.05 < p < 0.10)")
    else:
        print("Strategy is NOT statistically significant (p >= 0.10)")
        print("Performance could be explained by random chance.")
    
    plot_mcpt_results(results, metric_name="Profit Factor")

    sorted_splits = sorted(split_events, key=lambda x: x.date)
    n_train = int(len(sorted_splits) * 0.7)
    train_end_date = sorted_splits[n_train].date

    model, _ = train_split_model(
        split_events=sorted_splits[:n_train],
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=3,
        end_date=train_end_date
    )

    trades = simulate_trades(
        model=model,
        split_events=sorted_splits[n_train:],
        prices_by_symbol=prices_by_symbol,
        sector_momentum=sector_momentum,
        hold_duration=3,
        start_date=train_end_date
    )

    if trades:
        plot_trade_analysis(trades, title="Split Events Strategy - Trade Analysis")
