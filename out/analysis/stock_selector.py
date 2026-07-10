"""
Stock selector module - refactored with unified infrastructure.

Provides ML-based stock selection with:
- RPS (Relative Price Strength) filtering
- Strategy-based screening
- ML prediction using trained models
- Parallel processing for multiple stocks
"""

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.base import AnalyzerBase
from core.cache import cached, CacheConfig
from core.config import Config, get_config
from core.exceptions import AnalysisError, DataFetchError, ModelLoadError, ValidationError
from core.logger import get_logger
from core.parallel import ParallelProcessor
from utils.data_utils import clean_dataframe, validate_dataframe
from utils.date_utils import format_date, get_yesterday
from utils.validation import validate_range, validate_stock_code

logger = get_logger(__name__)


# ML library availability check
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    import joblib
    import yaml
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@dataclass
class StockPrediction:
    """
    Represents a stock prediction result.

    Attributes:
        stock_code: Stock code (e.g., "sh.600000")
        predicted_return: Predicted return rate (percentage)
        confidence: Prediction confidence (0-1)
        model_info: Information about the model used
        prediction_date: Date of prediction
        features_count: Number of features used
        data_points: Number of data points
    """
    stock_code: str
    predicted_return: float
    confidence: float
    model_info: str
    prediction_date: str
    features_count: int
    data_points: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stock_code": self.stock_code,
            "predicted_return_5d": self.predicted_return,
            "confidence": self.confidence,
            "model_info": self.model_info,
            "prediction_date": self.prediction_date,
            "features_count": self.features_count,
            "data_points": self.data_points,
        }

    @property
    def investment_score(self) -> float:
        """
        Calculate investment score.

        Returns:
            Investment score combining return and confidence
        """
        base_score = self.predicted_return * self.confidence

        if self.predicted_return < 0:
            base_score *= 0.5

        if self.confidence < 0.3:
            base_score *= 0.7
        elif self.confidence > 0.7:
            base_score *= 1.2

        return base_score


@dataclass
class SelectionCriteria:
    """
    Stock selection criteria.

    Attributes:
        rps5_threshold: Minimum RPS5 value (default: 85)
        prediction_days: Days to predict (default: 5)
        min_data_points: Minimum required data points (default: 50)
        enable_strategy_filter: Enable strategy-based filtering (default: True)
    """
    rps5_threshold: float = 85.0
    prediction_days: int = 5
    min_data_points: int = 50
    enable_strategy_filter: bool = True

    def __post_init__(self):
        """Validate criteria after initialization."""
        self.rps5_threshold = validate_range(
            self.rps5_threshold, 0, 100, "rps5_threshold"
        )
        self.prediction_days = validate_range(
            self.prediction_days, 1, 250, "prediction_days"
        )
        self.min_data_points = validate_range(
            self.min_data_points, 10, 1000, "min_data_points"
        )


@dataclass
class SelectionResult:
    """
    Stock selection result.

    Attributes:
        candidates: Candidate stock codes after RPS filtering
        strategy_selected: Stocks selected by strategies
        predictions: ML prediction results by stock code
        summary: Summary statistics
        timestamp: Result timestamp
    """
    candidates: List[str] = field(default_factory=list)
    strategy_selected: Dict[str, List[str]] = field(default_factory=dict)
    predictions: Dict[str, StockPrediction] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "step1_candidates": {
                "count": len(self.candidates),
                "stocks": self.candidates
            },
            "step2_strategy_selection": {
                strategy: {"count": len(stocks), "stocks": stocks}
                for strategy, stocks in self.strategy_selected.items()
            },
            "step3_ml_predictions": {
                code: pred.to_dict()
                for code, pred in self.predictions.items()
            },
            "summary": self.summary,
        }


class StockSelector(AnalyzerBase):
    """
    ML-based stock selector.

    Provides:
    - RPS-based candidate filtering
    - Strategy-based stock screening
    - ML prediction for future returns
    - Parallel processing for efficiency

    Example:
        selector = StockSelector(
            data_dir="market_data",
            models_dir="models"
        )
        result = selector.select_stocks(
            criteria=SelectionCriteria(rps5_threshold=85)
        )
        for stock, pred in result.predictions.items():
            print(f"{stock}: {pred.predicted_return:.2f}%")
    """

    def __init__(
        self,
        data_dir: str = "market_data",
        rps_dir: str = "rps_results",
        models_dir: str = "models",
        config: Optional[Config] = None,
    ):
        """
        Initialize stock selector.

        Args:
            data_dir: Directory containing stock data CSV files
            rps_dir: Directory containing RPS data files
            models_dir: Directory containing trained ML models
            config: Configuration (uses global config if None)
        """
        super().__init__(config)

        self.data_dir = data_dir
        self.rps_dir = rps_dir
        self.models_dir = models_dir

        # Data storage
        self._stock_data: Dict[str, pd.DataFrame] = {}
        self._rps_data: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Model storage
        self._loaded_models: Dict[str, Any] = {}
        self._loaded_scalers: Dict[str, Any] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}

        self.initialize()

    def initialize(self) -> None:
        """Initialize the selector."""
        super().initialize()

        if ML_AVAILABLE:
            self._load_models()

    def _load_models(self) -> None:
        """
        Load trained ML models and configurations.

        Raises:
            ModelLoadError: If model loading fails
        """
        if not os.path.exists(self.models_dir):
            self.logger.warning(f"Models directory not found: {self.models_dir}")
            return

        try:
            # Load model configurations
            config_files = glob.glob(os.path.join(self.models_dir, "*.yaml"))
            for config_file in config_files:
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                    config_name = os.path.basename(config_file).replace(".yaml", "")
                    self._model_configs[config_name] = config
                    self.logger.debug(f"Loaded config: {config_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load config {config_file}: {e}")

            # Load model files
            model_files = glob.glob(os.path.join(self.models_dir, "*.pkl"))
            for model_file in model_files:
                try:
                    model_name = os.path.basename(model_file).replace(".pkl", "")

                    if "scaler_" in model_name:
                        scaler_key = model_name.replace("scaler_", "")
                        self._loaded_scalers[scaler_key] = joblib.load(model_file)
                        self.logger.debug(f"Loaded scaler: {scaler_key}")
                    else:
                        self._loaded_models[model_name] = joblib.load(model_file)
                        self.logger.debug(f"Loaded model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model {model_file}: {e}")

            self.logger.info(
                f"Loaded {len(self._loaded_models)} models, "
                f"{len(self._loaded_scalers)} scalers, "
                f"{len(self._model_configs)} configs"
            )

        except Exception as e:
            raise ModelLoadError(f"Failed to load models: {e}")

    def _select_best_model(
        self,
        prediction_days: int = 5,
        task_type: str = "regression",
    ) -> Tuple[Any, Any, str]:
        """
        Select the best available model.

        Args:
            prediction_days: Prediction period in days
            task_type: Task type ("regression" or "classification")

        Returns:
            Tuple of (model, scaler, model_name)
        """
        if not self._loaded_models:
            return None, None, ""

        suitable_models = []

        for model_name in self._loaded_models.keys():
            if f"{prediction_days}d" in model_name:
                if task_type == "regression" and "return" in model_name and "gt_" not in model_name:
                    suitable_models.append(model_name)
                elif task_type == "classification" and ("gt_" in model_name or "multi_class" in model_name):
                    suitable_models.append(model_name)

        if not suitable_models:
            # Fallback to any model with matching days
            for model_name in self._loaded_models.keys():
                if f"{prediction_days}d" in model_name:
                    suitable_models.append(model_name)
                    break

        if not suitable_models:
            return None, None, ""

        # Prioritize random forest models
        for model_name in suitable_models:
            if "random_forest" in model_name:
                model = self._loaded_models[model_name]
                scaler = self._loaded_scalers.get(model_name)
                return model, scaler, model_name

        # Use first suitable model
        best_model_name = suitable_models[0]
        model = self._loaded_models[best_model_name]
        scaler = self._loaded_scalers.get(best_model_name)
        return model, scaler, best_model_name

    @cached(CacheConfig(ttl=3600))
    def load_stock_data(self, max_stocks: int = 100) -> int:
        """
        Load stock data from CSV files.

        Args:
            max_stocks: Maximum number of stocks to load

        Returns:
            Number of stocks successfully loaded

        Raises:
            DataFetchError: If data loading fails
        """
        if not os.path.exists(self.data_dir):
            raise DataFetchError(f"Data directory not found: {self.data_dir}")

        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))[:max_stocks]

        loaded_count = 0
        for csv_file in csv_files:
            try:
                stock_code = os.path.basename(csv_file).replace(".csv", "")
                df = pd.read_csv(csv_file)

                # Validate and clean data
                required_columns = ["date", "open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required_columns):
                    self.logger.debug(f"Stock {stock_code} missing required columns")
                    continue

                df = clean_dataframe(df, sort_by="date")
                validate_dataframe(df, required_columns, min_rows=30)

                if len(df) < 30:
                    self.logger.debug(f"Stock {stock_code} has insufficient data")
                    continue

                self._stock_data[stock_code] = df
                loaded_count += 1

            except Exception as e:
                self.logger.debug(f"Failed to load {csv_file}: {e}")
                continue

        self.logger.info(f"Loaded {loaded_count} stocks from {self.data_dir}")
        return loaded_count

    @cached(CacheConfig(ttl=3600))
    def load_rps_data(self) -> bool:
        """
        Load RPS (Relative Price Strength) data.

        Returns:
            True if successful or mock data was generated

        Raises:
            DataFetchError: If RPS data loading fails
        """
        if not os.path.exists(self.rps_dir):
            self.logger.warning(f"RPS directory not found: {self.rps_dir}, using mock data")
            return self._generate_mock_rps_data()

        rps_files = glob.glob(os.path.join(self.rps_dir, "RPS*.csv"))

        if not rps_files:
            self.logger.warning("No RPS files found, using mock data")
            return self._generate_mock_rps_data()

        loaded_periods = []

        for rps_file in rps_files:
            try:
                filename = os.path.basename(rps_file)

                # Determine period from filename
                period_map = {
                    "RPS5": "rps5",
                    "RPS10": "rps10",
                    "RPS20": "rps20",
                    "RPS60": "rps60",
                    "RPS120": "rps120",
                    "RPS250": "rps250",
                }

                period = None
                for key, value in period_map.items():
                    if key in filename:
                        period = value
                        break

                if not period:
                    continue

                df = pd.read_csv(rps_file)

                if not all(col in df.columns for col in ["stock_code", "date", "rps"]):
                    continue

                df["date"] = pd.to_datetime(df["date"])

                for stock_code, group in df.groupby("stock_code"):
                    if stock_code not in self._rps_data:
                        self._rps_data[stock_code] = {}

                    rps_dict = dict(zip(group["date"].dt.strftime("%Y-%m-%d"), group["rps"]))
                    self._rps_data[stock_code][period] = rps_dict

                loaded_periods.append(period)

            except Exception as e:
                self.logger.debug(f"Failed to load RPS file {rps_file}: {e}")
                continue

        if loaded_periods:
            self.logger.info(f"Loaded RPS periods: {loaded_periods}")
            return True

        return self._generate_mock_rps_data()

    def _generate_mock_rps_data(self) -> bool:
        """
        Generate mock RPS data for testing.

        Returns:
            True if successful
        """
        self.logger.info("Generating mock RPS data")

        for stock_code in self._stock_data.keys():
            dates = self._stock_data[stock_code].index.strftime("%Y-%m-%d").tolist()
            n = len(dates)

            # Generate mock RPS with proper ordering: RPS5 > RPS10 > RPS20 > RPS60 > RPS120 > RPS250
            base_rps = np.clip(np.random.normal(70, 15, n), 0, 100)

            self._rps_data[stock_code] = {
                "rps5": dict(zip(dates, np.clip(base_rps + 15, 0, 100))),
                "rps10": dict(zip(dates, np.clip(base_rps + 10, 0, 100))),
                "rps20": dict(zip(dates, np.clip(base_rps + 5, 0, 100))),
                "rps60": dict(zip(dates, np.clip(base_rps, 0, 100))),
                "rps120": dict(zip(dates, np.clip(base_rps - 5, 0, 100))),
                "rps250": dict(zip(dates, np.clip(base_rps - 10, 0, 100))),
            }

        return True

    def filter_by_rps(self, threshold: float = 85.0) -> List[str]:
        """
        Filter stocks by RPS5 threshold.

        Args:
            threshold: Minimum RPS5 value

        Returns:
            List of stock codes meeting the threshold
        """
        candidates = []

        for stock_code in self._stock_data.keys():
            if stock_code in self._rps_data and "rps5" in self._rps_data[stock_code]:
                rps5_dict = self._rps_data[stock_code]["rps5"]
                if rps5_dict:
                    latest_rps5 = rps5_dict[max(rps5_dict.keys())]
                    if latest_rps5 > threshold:
                        candidates.append(stock_code)

        self.logger.info(f"RPS5 > {threshold}: Found {len(candidates)} candidates")
        return candidates

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data.

        Args:
            df: Input dataframe with OHLCV data

        Returns:
            Dataframe with technical indicators
        """
        result = df.copy()

        # Moving averages
        for period in [5, 10, 20, 30, 60, 120]:
            result[f"ma{period}"] = df["close"].rolling(period).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(12).mean()
        ema26 = df["close"].ewm(26).mean()
        result["macd"] = ema12 - ema26
        result["macd_signal"] = result["macd"].ewm(9).mean()
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        # Bollinger Bands
        result["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        result["bb_upper"] = result["bb_middle"] + 2 * bb_std
        result["bb_lower"] = result["bb_middle"] - 2 * bb_std
        result["bb_position"] = (df["close"] - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])

        # Price position
        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()
        result["price_position_20"] = (df["close"] - low_20) / (high_20 - low_20)

        # Trend strength
        result["trend_strength"] = (result["ma20"] - result["ma60"]) / result["ma60"]

        # Momentum
        result["momentum_20"] = df["close"] / df["close"].shift(20) - 1

        # Returns
        for period in [1, 3, 5, 10, 20, 60]:
            result[f"return_{period}d"] = df["close"].pct_change(period)

        # Volatility
        for period in [10, 20, 60]:
            result[f"volatility_{period}d"] = df["close"].pct_change().rolling(period).std()

        # Volume ratio
        result["volume_ma"] = df["volume"].rolling(20).mean()
        result["volume_ratio"] = df["volume"] / result["volume_ma"]

        return result

    def _prepare_ml_features(
        self,
        df: pd.DataFrame,
        stock_code: str,
    ) -> pd.DataFrame:
        """
        Prepare ML features for prediction.

        Args:
            df: Stock data dataframe
            stock_code: Stock code (for RPS data lookup)

        Returns:
            Dataframe with 29 features for ML prediction
        """
        df_with_indicators = self._calculate_technical_indicators(df)

        # Add RPS data
        if stock_code in self._rps_data:
            rps_data = self._rps_data[stock_code]

            for idx in df_with_indicators.index:
                date_str = idx.strftime("%Y-%m-%d")

                for rps_period in ["rps5", "rps10", "rps20", "rps60"]:
                    if rps_period in rps_data and date_str in rps_data[rps_period]:
                        df_with_indicators.loc[idx, rps_period] = rps_data[rps_period][date_str]
                    else:
                        df_with_indicators.loc[idx, rps_period] = np.nan
        else:
            for rps_period in ["rps5", "rps10", "rps20", "rps60"]:
                df_with_indicators[rps_period] = 50.0

        # Define 29 features
        feature_columns = [
            "close", "ma5", "ma10", "ma20", "ma30", "ma60", "ma120",
            "price_position_20", "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_position", "volume_ratio", "trend_strength", "momentum_20",
            "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
            "volatility_10d", "volatility_20d", "volatility_60d",
            "rps5", "rps10", "rps20", "rps60"
        ]

        # Ensure all columns exist
        for col in feature_columns:
            if col not in df_with_indicators.columns:
                df_with_indicators[col] = 0.0

        return df_with_indicators[feature_columns]

    def _predict_single_stock(
        self,
        stock_code: str,
        prediction_days: int = 5,
    ) -> Optional[StockPrediction]:
        """
        Predict future return for a single stock.

        Args:
            stock_code: Stock code
            prediction_days: Days to predict

        Returns:
            StockPrediction if successful, None otherwise
        """
        if stock_code not in self._stock_data:
            return None

        df = self._stock_data[stock_code].copy()

        if len(df) < 50:
            return None

        try:
            # Prepare features
            features_df = self._prepare_ml_features(df, stock_code)

            if len(features_df) < 30:
                return None

            # Get numeric features
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
            features = features_df[numeric_columns].fillna(0)

            # Select best model
            model, scaler, model_name = self._select_best_model(
                prediction_days=prediction_days,
                task_type="regression"
            )

            if model is None:
                return None

            # Predict using latest features
            latest_features = features.iloc[[-1]]

            if scaler is not None:
                latest_features_scaled = scaler.transform(latest_features)
            else:
                latest_features_scaled = RobustScaler().fit_transform(features).iloc[[-1]]

            predicted_return = model.predict(latest_features_scaled)[0] * 100

            # Calculate confidence
            confidence = min(0.9, max(0.1, 1.0 - abs(predicted_return / 100) / 0.2))

            return StockPrediction(
                stock_code=stock_code,
                predicted_return=predicted_return,
                confidence=confidence,
                model_info=model_name or "unknown",
                prediction_date=format_date(get_yesterday()),
                features_count=len(features.columns),
                data_points=len(features),
            )

        except Exception as e:
            self.logger.warning(f"Prediction failed for {stock_code}: {e}")
            return None

    def predict_stocks(
        self,
        stock_codes: List[str],
        prediction_days: int = 5,
        parallel: bool = True,
    ) -> Dict[str, StockPrediction]:
        """
        Predict future returns for multiple stocks.

        Args:
            stock_codes: List of stock codes to predict
            prediction_days: Days to predict
            parallel: Use parallel processing

        Returns:
            Dictionary mapping stock codes to predictions
        """
        if not ML_AVAILABLE:
            raise AnalysisError("ML libraries not available")

        predictions = {}

        if parallel and len(stock_codes) > 10:
            # Use parallel processing
            max_workers = min(self.config.parallel.max_workers, len(stock_codes))

            with ParallelProcessor(max_workers=max_workers) as processor:
                results = processor.map(
                    lambda code: self._predict_single_stock(code, prediction_days),
                    stock_codes,
                    desc="Predicting stocks"
                )

            for result in results:
                if result is not None:
                    predictions[result.stock_code] = result
        else:
            # Sequential processing
            for stock_code in stock_codes:
                result = self._predict_single_stock(stock_code, prediction_days)
                if result is not None:
                    predictions[stock_code] = result

        self.logger.info(f"Predicted {len(predictions)} stocks")
        return predictions

    def select_stocks(
        self,
        criteria: Optional[SelectionCriteria] = None,
        max_stocks: int = 100,
    ) -> SelectionResult:
        """
        Run complete stock selection process.

        Args:
            criteria: Selection criteria
            max_stocks: Maximum stocks to load

        Returns:
            SelectionResult with candidates and predictions

        Raises:
            AnalysisError: If selection fails
        """
        criteria = criteria or SelectionCriteria()

        try:
            # Load data
            self.logger.info("Loading stock and RPS data")
            loaded = self.load_stock_data(max_stocks)
            if loaded == 0:
                raise AnalysisError("No stock data loaded")

            self.load_rps_data()

            # Filter by RPS
            candidates = self.filter_by_rps(criteria.rps5_threshold)

            if not candidates:
                self.logger.warning(f"No stocks found with RPS5 > {criteria.rps5_threshold}")
                return SelectionResult(summary={"total_candidates": 0})

            # Predict
            predictions = self.predict_stocks(
                candidates,
                prediction_days=criteria.prediction_days,
                parallel=True,
            )

            # Sort by investment score
            sorted_predictions = sorted(
                predictions.items(),
                key=lambda x: x[1].investment_score,
                reverse=True
            )

            result = SelectionResult(
                candidates=candidates,
                predictions=dict(sorted_predictions),
                summary={
                    "total_loaded_stocks": len(self._stock_data),
                    "total_candidates": len(candidates),
                    "total_predicted": len(predictions),
                }
            )

            return result

        except Exception as e:
            raise AnalysisError(f"Stock selection failed: {e}")

    def save_results(self, result: SelectionResult, filepath: str) -> bool:
        """
        Save selection results to file.

        Args:
            result: Selection result to save
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            self.logger.info(f"Results saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False

    def get_top_stocks(
        self,
        result: SelectionResult,
        top_n: int = 10,
    ) -> List[Tuple[str, StockPrediction]]:
        """
        Get top N stocks by investment score.

        Args:
            result: Selection result
            top_n: Number of top stocks to return

        Returns:
            List of (stock_code, prediction) tuples
        """
        sorted_predictions = sorted(
            result.predictions.items(),
            key=lambda x: x[1].investment_score,
            reverse=True
        )

        return sorted_predictions[:top_n]

    def print_report(self, result: SelectionResult, top_n: int = 10) -> None:
        """
        Print selection report.

        Args:
            result: Selection result
            top_n: Number of top stocks to display
        """
        print("\n" + "=" * 70)
        print("📊 Stock Selection Report")
        print("=" * 70)

        summary = result.summary
        print(f"\n📈 Summary:")
        print(f"  • Loaded stocks: {summary.get('total_loaded_stocks', 0)}")
        print(f"  • RPS candidates: {summary.get('total_candidates', 0)}")
        print(f"  • ML predictions: {summary.get('total_predicted', 0)}")

        top_stocks = self.get_top_stocks(result, top_n)

        if top_stocks:
            print(f"\n🏆 Top {len(top_stocks)} Stocks:")
            print("-" * 70)
            print(f"{'Rank':<5} {'Code':<12} {'Return':<10} {'Confidence':<12} {'Score':<10}")
            print("-" * 70)

            for i, (code, pred) in enumerate(top_stocks, 1):
                print(
                    f"{i:<5} {code:<12} {pred.predicted_return:>7.2f}% "
                    f"{pred.confidence:>11.3f} {pred.investment_score:>9.3f}"
                )

        print("=" * 70)


# Convenience functions for backward compatibility
def filter_stocks_by_rps(
    data_dir: str,
    rps_threshold: float = 85.0,
    max_stocks: int = 100,
) -> List[str]:
    """
    Filter stocks by RPS threshold.

    Args:
        data_dir: Stock data directory
        rps_threshold: RPS5 threshold
        max_stocks: Maximum stocks to load

    Returns:
        List of stock codes meeting the threshold
    """
    selector = StockSelector(data_dir=data_dir)
    selector.load_stock_data(max_stocks)
    selector.load_rps_data()
    return selector.filter_by_rps(rps_threshold)


def predict_stock_returns(
    stock_codes: List[str],
    data_dir: str = "market_data",
    prediction_days: int = 5,
) -> Dict[str, StockPrediction]:
    """
    Predict returns for given stocks.

    Args:
        stock_codes: List of stock codes
        data_dir: Stock data directory
        prediction_days: Days to predict

    Returns:
        Dictionary of predictions by stock code
    """
    selector = StockSelector(data_dir=data_dir)
    selector.load_stock_data()
    selector.load_rps_data()
    return selector.predict_stocks(stock_codes, prediction_days)
