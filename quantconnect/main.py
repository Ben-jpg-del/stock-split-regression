from AlgorithmImports import *
import numpy as np
import json


from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class SACRiskPolicy:
    def __init__(self, policy_json: str = None, policy_dict: dict = None):
        self.weights = None
        self.architecture = {}

        if policy_dict is not None:
            self._load_from_dict(policy_dict)
        elif policy_json is not None:
            policy_dict = json.loads(policy_json)
            self._load_from_dict(policy_dict)

    def _load_from_dict(self, policy_dict: dict):
        self.weights = {}
        for key, value in policy_dict['weights'].items():
            self.weights[key] = np.array(value, dtype=np.float32)
        self.architecture = policy_dict.get('architecture', {})

    def _relu(self, x):
        return np.maximum(0, x)

    def _tanh(self, x):
        return np.tanh(x)

    def get_position_multiplier(self, state: np.ndarray) -> float:
        if self.weights is None:
            return 1.0

        x = state.flatten().astype(np.float32)

        x = x @ self.weights['fc1.weight'].T + self.weights['fc1.bias']
        x = self._relu(x)

        x = x @ self.weights['fc2.weight'].T + self.weights['fc2.bias']
        x = self._relu(x)

        mean = x @ self.weights['mean_head.weight'].T + self.weights['mean_head.bias']

        action = self._tanh(mean)
        multiplier = (action + 1) / 2

        return float(np.clip(multiplier, 0, 1))

    def is_loaded(self) -> bool:
        return self.weights is not None



class SplitEventsAlgorithm(QCAlgorithm):
    BACKTEST_MODE = True
    USE_SAC_RISK_MANAGEMENT = False

    def initialize(self):
        self.set_cash(1_000)

        if self.BACKTEST_MODE:
            self.set_start_date(2023, 1, 1)
            self.set_end_date(2025, 12, 24)
        else:
            self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        self.universe_settings.resolution = Resolution.HOUR
        self.universe_settings.fill_forward = False
        self.universe_settings.asynchronous = True
        self.universe_settings.data_normalization_mode = DataNormalizationMode.Raw
        self._universe = self.add_universe(
            lambda fundamental: [
                x.symbol 
                for x in fundamental 
                if (x.asset_classification.morningstar_sector_code == 
                    MorningstarSectorCode.TECHNOLOGY)
            ]
        )

        self._max_open_trades = self.get_parameter('max_open_trades', 4)
        self._hold_duration = timedelta(self.get_parameter('hold_duration', 3))
        self._training_lookback = timedelta(
            self.get_parameter('training_lookback_years', 4) * 365
        )

        self._min_order_value = self.get_parameter('min_order_value', 50)
        self._dry_run = self.get_parameter('dry_run', False)
        
        self._sector_etf = self.add_equity(
            "XLK", self.universe_settings.resolution
        )
        self._sector_etf.roc = self.roc(
            self._sector_etf.symbol, 22, Resolution.DAILY
        )
        self._sector_etf.roc_history = pd.Series()
        self._sector_etf.roc.updated += self._update_event_handler
        bars = self.history[TradeBar](
            self._sector_etf.symbol, 
            self._training_lookback.days + self._sector_etf.roc.warm_up_period, 
            Resolution.DAILY
        )
        for bar in bars:
            self._sector_etf.roc.update(bar.end_time, bar.close)
    
        self._target_exposure_per_trade = 1 / self._max_open_trades
        self._trades_by_symbol = {}
        self._model = LinearRegression()

        self.train(
            self.date_rules.month_start(self._sector_etf.symbol),
            self.time_rules.midnight,
            self._train
        )

        if self.BACKTEST_MODE:
            self.schedule.on(
                self.date_rules.on(self.start_date.year, self.start_date.month, self.start_date.day + 1),
                self.time_rules.after_market_open(self._sector_etf.symbol, 1),
                self._train
            )
        else:   
            self.set_warmup(timedelta(days=5))

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.midnight,
            self._scan_for_trade_exits
        )

        self._sac_policy = None
        self._sac_policy_key = self.get_parameter('sac_policy_key', 'policy_export_compact.json')
        if self.USE_SAC_RISK_MANAGEMENT:
            try:
                if self.object_store.contains_key(self._sac_policy_key):
                    policy_json = self.object_store.read(self._sac_policy_key)
                    self._sac_policy = SACRiskPolicy(policy_json=policy_json)
                    self.log(f"SAC Risk Policy loaded from Object Store: {self._sac_policy_key}")
                else:
                    self.log(f"SAC policy not found in Object Store: {self._sac_policy_key}")
                    self.log("Using default risk management (no SAC)")
                    self._sac_policy = SACRiskPolicy()
            except Exception as e:
                self.log(f"Failed to load SAC policy: {e}")
                self._sac_policy = SACRiskPolicy()

    def on_warmup_finished(self):
        if not self.BACKTEST_MODE:
            self.log("Warmup finished, training model on historical data...")
            self._train()

    def _update_event_handler(self, indicator, indicator_data_point):
        if not indicator.is_ready:
            return
        t = indicator_data_point.end_time
        self._sector_etf.roc_history.loc[t] = indicator_data_point.value
        self._sector_etf.roc_history = self._sector_etf.roc_history[
            self._sector_etf.roc_history.index > t - self._training_lookback
        ]

    def _train(self):
        if not self._universe.selected:
            self.log("Universe not populated yet, skipping training")
            return

        splits = self.history[Split](
            self._universe.selected, self._training_lookback
        )
        assets_with_splits = set([])
        for splits_dict in splits:
            for symbol in splits_dict.keys():
                assets_with_splits.add(symbol)
        prices = self.history(
            list(assets_with_splits), self._training_lookback, Resolution.DAILY, 
            data_normalization_mode=DataNormalizationMode.SCALED_RAW
        )['open'].unstack(0)
        
        samples = np.empty((0, 3))
        for splits_dict in splits:
            for symbol, split in splits_dict.items():
                if split.type == SplitType.SPLIT_OCCURRED:
                    continue
                t = split.end_time
                entry_series = prices[symbol].loc[t < prices.index]
                if entry_series.empty or np.isnan(entry_series[0]):
                    continue
                entry_price = entry_series[0]
                exit_series = prices[symbol].loc[
                    t + self._hold_duration < prices.index
                ]
                if exit_series.empty or np.isnan(exit_series[0]):
                    continue
                exit_price = exit_series[0]
                filtered_roc = self._sector_etf.roc_history[
                    self._sector_etf.roc_history.index <= t
                ]
                if filtered_roc.empty:
                    continue
                sector_roc = filtered_roc.iloc[-1]
                sample = np.array([
                    split.split_factor,
                    sector_roc,
                    (exit_price - entry_price) / entry_price
                ])
                samples = np.append(samples, [sample], axis=0)

        self.plot("Samples", "Count", len(samples))
        self._model.fit(samples[:, :2], samples[:, -1])

    def on_splits(self, splits):
        for symbol, split in splits.items():
            if symbol == self._sector_etf.symbol:
                continue
            
            if (split.type == SplitType.WARNING and
                sum(
                    [len(trades) for trades in self._trades_by_symbol.values()]
                ) < self._max_open_trades):
                try:
                    check_is_fitted(self._model)
                except NotFittedError:
                    self.log(f"Model not fitted yet, skipping split for {symbol}")
                    continue

                factors = [
                    split.split_factor, self._sector_etf.roc.current.value
                ]
                predicted_return = self._model.predict([factors])[0]
                self.log(f"{self.time};{str(symbol.id)};{predicted_return}")
                if predicted_return == 0:
                    continue

                if symbol not in self._trades_by_symbol:
                    self._trades_by_symbol[symbol] = []
                quantity = self.calculate_order_quantity(
                    symbol,
                    np.sign(predicted_return) * self._target_exposure_per_trade
                )
                if quantity == 0:
                    continue

                if self._sac_policy is not None and self._sac_policy.is_loaded():
                    quantity = self._apply_sac_risk_adjustment(
                        symbol, quantity, predicted_return
                    )
                    if quantity == 0:
                        self.log(f"SAC Risk: Skipping {symbol} (multiplier reduced to 0)")
                        continue

                security = self.securities[symbol]
                order_value = abs(quantity) * security.price
                if order_value < self._min_order_value:
                    self.log(f"Order value ${order_value:.2f} below minimum ${self._min_order_value}, skipping {symbol}")
                    continue

                self._trades_by_symbol[symbol].append(
                    Trade(self, symbol, self._hold_duration, quantity, self._dry_run)
                )

            elif (split.type == SplitType.SPLIT_OCCURRED and 
                symbol in self._trades_by_symbol):
                for trade in self._trades_by_symbol[symbol]:
                    trade.on_split_occurred(split)

    def _scan_for_trade_exits(self):
        closed_trades = []
        for trades in self._trades_by_symbol.values():
            closed_trades = []
            for i, trade in enumerate(trades):
                trade.scan(self)
                if trade.closed:
                    closed_trades.append(i)

            for i in closed_trades[::-1]:
                del trades[i]

    def _get_risk_state(self, symbol, quantity, predicted_return):
        security = self.securities[symbol]
        portfolio_value = max(self.portfolio.total_portfolio_value, 1.0)

        direction = 1.0 if quantity > 0 else -1.0 if quantity < 0 else 0.0
        position_value = abs(quantity) * security.price
        size_pct = position_value / portfolio_value

        days_held = 0.0
        sector_roc = self._sector_etf.roc.current.value if self._sector_etf.roc.is_ready else 0.0

        volatility_ratio = 1.0 
        price_vs_sma = 0.0
        margin_used = self.portfolio.total_margin_used
        margin_remaining = self.portfolio.margin_remaining
        margin_usage = margin_used / (margin_used + margin_remaining) if (margin_used + margin_remaining) > 0 else 0.0

        total_holdings_value = sum(
            abs(h.holdings_value) for h in self.portfolio.values() if h.invested
        )
        portfolio_heat = total_holdings_value / portfolio_value if portfolio_value > 0 else 0.0

        state = np.array([
            direction,          
            size_pct,           
            days_held / 3.0,     
            0.0,                 
            0.0,                 
            0.0,                 
            sector_roc,          
            volatility_ratio,    
            price_vs_sma,        
            margin_usage,        
            portfolio_heat,      
        ], dtype=np.float32)

        return state

    def _apply_sac_risk_adjustment(self, symbol, quantity, predicted_return):
        if self._sac_policy is None or quantity == 0:
            return quantity

        try:
            state = self._get_risk_state(symbol, quantity, predicted_return)

            position_multiplier = self._sac_policy.get_position_multiplier(state)

            adjusted_quantity = int(quantity * position_multiplier)

            if position_multiplier < 0.5:
                self.log(f"SAC Risk: {symbol} reduced from {quantity} to {adjusted_quantity} "
                        f"(multiplier={position_multiplier:.2f})")

            return adjusted_quantity

        except Exception as e:
            self.log(f"SAC risk adjustment failed: {e}")
            return quantity


class Trade:
    def __init__(self, algorithm, symbol, hold_duration, quantity, dry_run=False):
        self.closed = False
        self._symbol = symbol
        self._close_time = algorithm.time + hold_duration
        self._quantity = quantity
        self._dry_run = dry_run
        if dry_run:
            algorithm.log(f"[DRY RUN] Would open: {symbol} qty={quantity}")
        else:
            algorithm.market_on_open_order(symbol, quantity)

    def on_split_occurred(self, split):
        self._quantity = int(self._quantity / split.split_factor)

    def scan(self, algorithm):
        if not self.closed and self._close_time <= algorithm.time:
            if self._dry_run:
                algorithm.log(f"[DRY RUN] Would close: {self._symbol} qty={-self._quantity}")
            else:
                algorithm.market_on_open_order(self._symbol, -self._quantity)
            self.closed = True