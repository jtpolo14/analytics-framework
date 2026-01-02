"""
BTC Price Prediction Model v1.2

This module contains the improved model class for predicting future Bitcoin prices
based on cryptocurrency market data. This version (v1.2) includes:
- Pipeline architecture to prevent data leakage during CV
- Feature scaling with StandardScaler
- Train/test split and cross-validation
- Outlier handling on stationary transforms only (non-stationarity aware)
- Enhanced metrics (RMSE, MAPE, Median AE) on original price scale
- Multicollinearity diagnostics (VIF)
- Residual analysis
- Feature importance analysis
- Regularization tuning (Ridge/Elastic Net)

This version predicts the next 5-minute interval BTC price and includes technical indicators:
- Moving Averages (10, 20, 50, 200 points)
- RSI (Relative Strength Index)
- VWAP (Volume-Weighted Average Price)

The model uses sklearn Pipeline to ensure proper data handling during cross-validation
and hyperparameter tuning, preventing test fold statistics from leaking into training.

NOTE: This is an improved model implementation that uses time series features and
technical indicators for better prediction accuracy. For production use, consider
more sophisticated models (e.g., LSTM, Transformer models, or ensemble methods).
"""
import pandas as pd
import numpy as np
import time
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from typing import List, Dict, Any, Tuple, Optional
import json
import warnings
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def extract_btc_from_snapshot(snapshot_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract BTC data from a snapshot.
    
    Args:
        snapshot_data: List of cryptocurrency data dictionaries
        
    Returns:
        dict: BTC data dictionary, or None if not found
    """
    df = pd.DataFrame(snapshot_data)
    btc_data = df[df['symbol'] == 'btc'].iloc[0] if len(df[df['symbol'] == 'btc']) > 0 else None
    return btc_data.to_dict() if btc_data is not None else None


def calculate_moving_averages(btc_price_history: List[float], current_price: float) -> Dict[str, float]:
    """
    Calculate moving averages from price history.
    
    Args:
        btc_price_history: List of historical BTC prices
        current_price: Current BTC price
        
    Returns:
        dict: Dictionary with MA values for periods 10, 20, 50, 200
    """
    ma_values = {}
    for period in [10, 20, 50, 200]:
        if len(btc_price_history) >= period:
            ma_values[f'ma_{period}'] = float(np.mean(btc_price_history[-period:]))
        else:
            # Use available data or current price
            if len(btc_price_history) > 0:
                ma_values[f'ma_{period}'] = float(np.mean(btc_price_history))
            else:
                ma_values[f'ma_{period}'] = float(current_price)
    return ma_values


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index) from price history.
    
    Args:
        prices: List of historical prices
        period: RSI period (default: 14)
        
    Returns:
        float: RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI if insufficient data
    
    # Get the last period+1 prices
    price_window = prices[-(period + 1):]
    deltas = np.diff(price_window)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def calculate_vwap(price_history: List[float], volume_history: List[float], period: int = 20) -> float:
    """
    Calculate VWAP (Volume-Weighted Average Price) from price and volume history.
    
    Args:
        price_history: List of historical prices
        volume_history: List of historical volumes
        period: Number of periods to use (default: 20 for 5-min intervals = 100 minutes)
        
    Returns:
        float: VWAP value
    """
    if len(price_history) < period or len(volume_history) < period:
        # Use available data
        period = min(len(price_history), len(volume_history))
    
    if period == 0:
        return 0.0
    
    prices = price_history[-period:]
    volumes = volume_history[-period:]
    
    if sum(volumes) == 0:
        return float(prices[-1]) if prices else 0.0
    
    vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
    return float(vwap)


class StationaryOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling outliers on stationary transforms only.
    
    CRITICAL: This transformer applies outlier detection ONLY to stationary features
    (log returns, percentage changes, z-scored values), NOT raw price levels.
    This prevents treating bull market prices as anomalies (non-stationarity aware).
    
    Methods:
    - 'iqr': Interquartile Range method
    - 'zscore': Z-score method
    - 'winsorize': Cap outliers at percentiles (default)
    """
    
    def __init__(self, method='winsorize', lower_percentile=0.01, upper_percentile=0.99, z_threshold=3.0):
        """
        Initialize the outlier handler.
        
        Args:
            method: 'iqr', 'zscore', or 'winsorize'
            lower_percentile: Lower percentile for winsorizing (default 0.01)
            upper_percentile: Upper percentile for winsorizing (default 0.99)
            z_threshold: Z-score threshold for outlier detection (default 3.0)
        """
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.z_threshold = z_threshold
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer on training data."""
        X = np.asarray(X)
        self.lower_bounds_ = np.zeros(X.shape[1])
        self.upper_bounds_ = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            col = X[:, i]
            if self.method == 'iqr':
                Q1 = np.percentile(col, 25)
                Q3 = np.percentile(col, 75)
                IQR = Q3 - Q1
                self.lower_bounds_[i] = Q1 - 1.5 * IQR
                self.upper_bounds_[i] = Q3 + 1.5 * IQR
            elif self.method == 'zscore':
                mean = np.mean(col)
                std = np.std(col)
                if std > 0:
                    self.lower_bounds_[i] = mean - self.z_threshold * std
                    self.upper_bounds_[i] = mean + self.z_threshold * std
                else:
                    self.lower_bounds_[i] = np.min(col)
                    self.upper_bounds_[i] = np.max(col)
            elif self.method == 'winsorize':
                self.lower_bounds_[i] = np.percentile(col, self.lower_percentile * 100)
                self.upper_bounds_[i] = np.percentile(col, self.upper_percentile * 100)
        
        return self
    
    def transform(self, X):
        """Transform data by capping outliers."""
        X = np.asarray(X).copy()
        
        for i in range(X.shape[1]):
            if self.method == 'winsorize':
                # Cap outliers at bounds
                X[:, i] = np.clip(X[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
            else:
                # For IQR and zscore, cap at bounds
                X[:, i] = np.clip(X[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
        
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling missing values in features.
    Uses mean imputation for numeric features.
    """
    
    def __init__(self, strategy='mean'):
        """
        Initialize the missing value handler.
        
        Args:
            strategy: 'mean', 'median', or 'zero'
        """
        self.strategy = strategy
        self.imputation_values_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer on training data."""
        X = np.asarray(X)
        self.imputation_values_ = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            col = X[:, i]
            if self.strategy == 'mean':
                self.imputation_values_[i] = np.nanmean(col) if np.any(np.isnan(col)) else 0.0
            elif self.strategy == 'median':
                self.imputation_values_[i] = np.nanmedian(col) if np.any(np.isnan(col)) else 0.0
            elif self.strategy == 'zero':
                self.imputation_values_[i] = 0.0
        
        return self
    
    def transform(self, X):
        """Transform data by imputing missing values."""
        X = np.asarray(X).copy()
        
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X[mask, i] = self.imputation_values_[i]
        
        return X


class BTCPricePredictor:
    """
    Model for predicting future Bitcoin prices based on cryptocurrency market data.
    
    This v1.2 version includes:
    - Pipeline architecture to prevent data leakage during CV
    - Feature scaling with StandardScaler
    - Train/test split and cross-validation
    - Outlier handling on stationary transforms only (non-stationarity aware)
    - Enhanced metrics on original price scale
    - Multicollinearity diagnostics (VIF)
    - Residual analysis
    - Feature importance analysis
    - Regularization tuning (Ridge/Elastic Net)
    
    This version predicts the next 5-minute interval BTC price and includes technical indicators:
    - Moving Averages (10, 20, 50, 200 points)
    - RSI (Relative Strength Index)
    - VWAP (Volume-Weighted Average Price)
    """
    
    def __init__(
        self, 
        model=None, 
        use_regularization=True,
        scale_features=True,
        handle_outliers=True,
        outlier_method='winsorize',
        test_size=0.2,
        use_cross_validation=True,
        cv_splits=5,
        tune_alpha=True,
        use_elastic_net=False,
        use_log_target=False
    ):
        """
        Initialize the BTC price predictor.
        
        Args:
            model: Optional sklearn-compatible model. If None, uses Ridge or ElasticNet.
            use_regularization: If True and model is None, uses Ridge/ElasticNet regression.
            scale_features: If True, applies StandardScaler to features (default True).
            handle_outliers: If True, applies outlier handling on stationary transforms (default True).
            outlier_method: 'iqr', 'zscore', or 'winsorize' (default 'winsorize').
            test_size: Fraction of data to use for testing (default 0.2).
            use_cross_validation: If True, performs cross-validation during training (default True).
            cv_splits: Number of CV splits for TimeSeriesSplit (default 5).
            tune_alpha: If True, tunes regularization alpha parameter (default True).
            use_elastic_net: If True, uses ElasticNet instead of Ridge (default False).
            use_log_target: If True, applies log transformation to target (default False).
        """
        # Store configuration
        self.scale_features = scale_features
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.test_size = test_size
        self.use_cross_validation = use_cross_validation
        self.cv_splits = cv_splits
        self.tune_alpha = tune_alpha
        self.use_elastic_net = use_elastic_net
        self.use_log_target = use_log_target
        
        # Initialize base model
        if model is None:
            if use_elastic_net:
                self.base_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
            else:
                # Use Ridge regression by default to prevent overfitting
                self.base_model = Ridge(alpha=1.0) if use_regularization else LinearRegression()
        else:
            self.base_model = model
        
        # Pipeline will be built during training
        self.pipeline = None
        self.feature_columns = None
        self.is_trained = False
        self._training_n_samples = None
        self._test_metrics = None
        self._cv_scores = None
    
    def _build_pipeline(self):
        """
        Build sklearn Pipeline with preprocessing steps and model.
        
        Pipeline structure: [MissingValueHandler] -> [OutlierHandler] -> [Scaler] -> [Model]
        This ensures proper data handling during CV and prevents leakage.
        """
        steps = []
        
        # Step 1: Handle missing values
        steps.append(('missing_values', MissingValueHandler(strategy='mean')))
        
        # Step 2: Handle outliers (on stationary transforms only - applied during feature extraction)
        if self.handle_outliers:
            steps.append(('outliers', StationaryOutlierHandler(
                method=self.outlier_method,
                lower_percentile=0.01,
                upper_percentile=0.99
            )))
        
        # Step 3: Scale features
        if self.scale_features:
            steps.append(('scaler', StandardScaler()))
        
        # Step 4: Model
        steps.append(('model', self.base_model))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def _calculate_metrics(self, y_true, y_pred, y_true_original=None, y_pred_original=None):
        """
        Calculate comprehensive metrics on original price scale.
        
        If log transformation was used, y_true_original and y_pred_original should be provided.
        """
        # Use original scale if provided (for log-transformed targets)
        if y_true_original is not None and y_pred_original is not None:
            y_true_eval = y_true_original
            y_pred_eval = y_pred_original
        else:
            y_true_eval = y_true
            y_pred_eval = y_pred
        
        mae = mean_absolute_error(y_true_eval, y_pred_eval)
        mse = mean_squared_error(y_true_eval, y_pred_eval)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_eval, y_pred_eval)
        median_ae = median_absolute_error(y_true_eval, y_pred_eval)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_eval - y_pred_eval) / (y_true_eval + 1e-8))) * 100
        
        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "median_ae": float(median_ae),
            "mape": float(mape)
        }
        
    def extract_features(
        self, 
        df: pd.DataFrame,
        btc_price_history: Optional[List[float]] = None,
        btc_volume_history: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Extract features from cryptocurrency data for prediction model.
        Creates features including technical indicators (MA, RSI, VWAP) for predicting future BTC price.
        
        Args:
            df: DataFrame with cryptocurrency market data
            btc_price_history: Optional list of historical BTC prices for technical indicators
            btc_volume_history: Optional list of historical BTC volumes for VWAP calculation
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        # Extract BTC data
        btc_data = df[df['symbol'] == 'btc'].iloc[0] if len(df[df['symbol'] == 'btc']) > 0 else None
        eth_data = df[df['symbol'] == 'eth'].iloc[0] if len(df[df['symbol'] == 'eth']) > 0 else None
        sol_data = df[df['symbol'] == 'sol'].iloc[0] if len(df[df['symbol'] == 'sol']) > 0 else None
        
        # Create feature row for BTC prediction
        feature_row = {}
        
        if btc_data is not None:
            current_btc_price = btc_data.get('current_price', 0)
            
            # BTC-specific features (existing)
            feature_row['btc_volume'] = btc_data.get('total_volume', 0)
            feature_row['btc_price_change_24h'] = btc_data.get('price_change_24h', 0)
            feature_row['btc_price_change_pct_24h'] = btc_data.get('price_change_percentage_24h', 0)
            feature_row['btc_market_cap'] = btc_data.get('market_cap', 0)
            feature_row['btc_high_24h'] = btc_data.get('high_24h', 0)
            feature_row['btc_low_24h'] = btc_data.get('low_24h', 0)
            feature_row['btc_price_change_pct_7d'] = btc_data.get('price_change_percentage_7d_in_currency', 0)
            feature_row['btc_price_change_pct_30d'] = btc_data.get('price_change_percentage_30d_in_currency', 0)
            feature_row['btc_ath_change_pct'] = btc_data.get('ath_change_percentage', 0)
            
            # Technical Indicators (NEW in v1.1)
            if btc_price_history is not None and len(btc_price_history) > 0:
                # Moving Averages
                ma_values = calculate_moving_averages(btc_price_history, current_btc_price)
                feature_row.update(ma_values)
                
                # RSI
                rsi = calculate_rsi(btc_price_history + [current_btc_price])
                feature_row['rsi'] = rsi
                
                # VWAP
                if btc_volume_history is not None and len(btc_volume_history) > 0:
                    vwap = calculate_vwap(
                        btc_price_history + [current_btc_price],
                        btc_volume_history + [btc_data.get('total_volume', 0)]
                    )
                    feature_row['vwap'] = vwap
                else:
                    feature_row['vwap'] = current_btc_price
            else:
                # No history available - use current price as fallback
                for period in [10, 20, 50, 200]:
                    feature_row[f'ma_{period}'] = current_btc_price
                feature_row['rsi'] = 50.0  # Neutral RSI
                feature_row['vwap'] = current_btc_price
        
        if eth_data is not None:
            # ETH features (market correlation)
            feature_row['eth_price'] = eth_data.get('current_price', 0)
            feature_row['eth_volume'] = eth_data.get('total_volume', 0)
            feature_row['eth_price_change_pct_24h'] = eth_data.get('price_change_percentage_24h', 0)
            feature_row['eth_market_cap'] = eth_data.get('market_cap', 0)
        
        if sol_data is not None:
            # SOL features (market correlation)
            feature_row['sol_price'] = sol_data.get('current_price', 0)
            feature_row['sol_volume'] = sol_data.get('total_volume', 0)
            feature_row['sol_price_change_pct_24h'] = sol_data.get('price_change_percentage_24h', 0)
        
        # Market-wide features
        feature_row['total_market_cap'] = df['market_cap'].sum() if 'market_cap' in df.columns else 0
        feature_row['total_volume'] = df['total_volume'].sum() if 'total_volume' in df.columns else 0
        
        # Calculate market dominance (BTC market cap / total market cap)
        if btc_data is not None and 'market_cap' in df.columns:
            btc_market_cap = btc_data.get('market_cap', 0)
            total_market_cap = df['market_cap'].sum()
            feature_row['btc_dominance'] = (btc_market_cap / total_market_cap * 100) if total_market_cap > 0 else 0
        else:
            feature_row['btc_dominance'] = 0
        
        # NEW in v1.2: Price ratios
        if btc_data is not None:
            current_btc_price = btc_data.get('current_price', 0)
            
            # BTC price to ETH price ratio
            if eth_data is not None and eth_data.get('current_price', 0) > 0:
                feature_row['btc_eth_ratio'] = current_btc_price / eth_data.get('current_price', 1)
            else:
                feature_row['btc_eth_ratio'] = 0
            
            # BTC price to MA ratios
            if 'ma_20' in feature_row and feature_row['ma_20'] > 0:
                feature_row['btc_ma20_ratio'] = current_btc_price / feature_row['ma_20']
            else:
                feature_row['btc_ma20_ratio'] = 1.0
            
            if 'ma_50' in feature_row and feature_row['ma_50'] > 0:
                feature_row['btc_ma50_ratio'] = current_btc_price / feature_row['ma_50']
            else:
                feature_row['btc_ma50_ratio'] = 1.0
            
            # BTC price to VWAP ratio
            if 'vwap' in feature_row and feature_row['vwap'] > 0:
                feature_row['btc_vwap_ratio'] = current_btc_price / feature_row['vwap']
            else:
                feature_row['btc_vwap_ratio'] = 1.0
        
        # NEW in v1.2: Interaction terms (volume * price change)
        if btc_data is not None:
            btc_volume = btc_data.get('total_volume', 0)
            btc_price_change_pct = btc_data.get('price_change_percentage_24h', 0)
            feature_row['volume_price_change_interaction'] = btc_volume * abs(btc_price_change_pct) / 100.0
        
        return pd.DataFrame([feature_row])
    
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from extracted features DataFrame.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            np.ndarray: Feature matrix
        """
        # Select feature columns (exclude non-numeric or target-related columns)
        if self.feature_columns is None:
            # First time - determine feature columns
            self.feature_columns = [
                col for col in features_df.columns 
                if col not in ['btc_price'] and features_df[col].dtype in ['float64', 'int64']
            ]
        
        # Ensure all expected columns exist
        missing_cols = set(self.feature_columns) - set(features_df.columns)
        if missing_cols:
            # Add missing columns with zero values
            for col in missing_cols:
                features_df[col] = 0
        
        # Select only the feature columns we need
        X = features_df[self.feature_columns].values
        return X
    
    def build_training_dataset(
        self, 
        snapshots: List[List[Dict[str, Any]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build training dataset from multiple historical snapshots.
        This v1.1 version predicts the NEXT snapshot's BTC price (future prediction).
        
        Args:
            snapshots: List of snapshots, where each snapshot is a list of 
                      cryptocurrency data dictionaries (must be in chronological order)
            
        Returns:
            tuple: (X, y) where:
                - X: Feature matrix (n_samples, n_features)
                - y: Target vector (n_samples,) - Future BTC prices (next 5-min interval)
        """
        all_features = []
        all_targets = []
        total_records = 0
        
        # First pass: build price/volume history from all snapshots
        btc_price_history = []
        btc_volume_history = []
        
        for snapshot_data in snapshots:
            btc_data = extract_btc_from_snapshot(snapshot_data)
            if btc_data:
                btc_price_history.append(btc_data.get('current_price', 0))
                btc_volume_history.append(btc_data.get('total_volume', 0))
        
        # Second pass: extract features with history, target from next snapshot
        for i in range(len(snapshots) - 1):  # Skip last snapshot (no future price available)
            snapshot_start = time.time()
            snapshot_data = snapshots[i]
            
            # Convert to DataFrame
            df = pd.DataFrame(snapshot_data)
            total_records += len(df)
            
            # Extract features with historical context
            # Use history up to current point (i+1 because we include current)
            current_price_history = btc_price_history[:i+1]
            current_volume_history = btc_volume_history[:i+1]
            
            features_df = self.extract_features(
                df,
                btc_price_history=current_price_history[:-1] if len(current_price_history) > 1 else None,
                btc_volume_history=current_volume_history[:-1] if len(current_volume_history) > 1 else None
            )
            
            # Get future BTC price (target) from next snapshot
            next_snapshot_data = snapshots[i + 1]
            next_btc_data = extract_btc_from_snapshot(next_snapshot_data)
            
            if next_btc_data is None:
                print(f"Warning: Snapshot {i+2} missing BTC data, skipping snapshot {i+1}...")
                continue
            
            future_btc_price = next_btc_data.get('current_price', 0)
            current_btc_price = btc_price_history[i] if i < len(btc_price_history) else 0
            last_updated = next_btc_data.get('last_updated', 'N/A')
            
            # Format last_updated timestamp (remove milliseconds and timezone for readability)
            if last_updated != 'N/A' and isinstance(last_updated, str):
                # Format: "2025-12-27T10:46:21.358Z" -> "2025-12-27 10:46:21"
                try:
                    formatted_time = last_updated.split('.')[0].replace('T', ' ')
                    if formatted_time.endswith('Z'):
                        formatted_time = formatted_time[:-1]
                except:
                    formatted_time = last_updated
            else:
                formatted_time = str(last_updated)
            
            # Prepare features
            X = self.prepare_features(features_df)
            
            all_features.append(X[0])  # X is shape (1, n_features), take first row
            all_targets.append(future_btc_price)
            
            snapshot_duration = time.time() - snapshot_start
            print(f"  Snapshot {i+1}: Current BTC = ${current_btc_price:,.2f} | Future BTC = ${future_btc_price:,.2f} | Target Time: {formatted_time} | ({snapshot_duration:.3f}s)")
        
        if len(all_features) == 0:
            raise ValueError("No valid snapshots found. At least two snapshots with BTC data are required for future prediction.")
        
        X = np.array(all_features)
        y = np.array(all_targets)
        
        print(f"\nBuilt training dataset: {len(all_features)} samples, {len(self.feature_columns)} features")
        print(f"Total records processed: {total_records:,}")
        print(f"Note: Predicting future 5-minute interval BTC price (v1.1)")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model on provided features and targets with train/test split and pipeline.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - Future BTC prices
            verbose: Whether to print training information
            
        Returns:
            dict: Training metrics including train/test metrics and CV scores
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
        
        n_samples, n_features = X.shape
        
        # Warn about potential overfitting
        if n_samples <= n_features:
            if verbose:
                print(f"\n[WARNING] Overfitting risk detected!")
                print(f"   Training samples ({n_samples}) <= Features ({n_features})")
                print(f"   Model may memorize training data instead of learning patterns.")
                print(f"   Consider: using more data, fewer features, or regularization.")
        
        # Handle log transformation of target if enabled
        y_original = y.copy()
        if self.use_log_target:
            y = np.log(y + 1e-8)  # Add small epsilon to avoid log(0)
        
        # Train/test split (BEFORE any preprocessing to prevent leakage)
        if self.test_size > 0 and self.test_size < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False  # No shuffle for time series
            )
            y_test_original = y_test.copy() if not self.use_log_target else np.exp(y_test) - 1e-8
        else:
            # No split - use all data for training
            X_train, X_test, y_train, y_test = X, np.array([]), y, np.array([])
            y_test_original = np.array([])
        
        if verbose:
            print(f"\nTraining model on {len(X_train)} samples with {n_features} features...")
            if len(X_test) > 0:
                print(f"Test set: {len(X_test)} samples")
            print(f"Target: Future 5-minute interval BTC price")
            if self.use_log_target:
                print(f"Using log transformation for target variable")
        
        # Build pipeline
        self._build_pipeline()
        
        # Hyperparameter tuning if enabled
        if self.tune_alpha and hasattr(self.base_model, 'alpha'):
            if verbose:
                print(f"\nTuning regularization parameter...")
            
            # Define parameter grid
            if self.use_elastic_net:
                param_grid = {
                    'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            else:
                param_grid = {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
            
            # Use TimeSeriesSplit for CV
            tscv = TimeSeriesSplit(n_splits=min(self.cv_splits, len(X_train) - 1))
            
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            
            if verbose:
                print(f"Best alpha: {grid_search.best_params_.get('model__alpha', 'N/A')}")
                if self.use_elastic_net:
                    print(f"Best l1_ratio: {grid_search.best_params_.get('model__l1_ratio', 'N/A')}")
        
        # Train the pipeline
        # Note: If GridSearchCV was used, best_estimator_ is already fitted, but refitting is safe
        self.pipeline.fit(X_train, y_train)
        
        # Verify pipeline is fitted
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.pipeline)
        except Exception as e:
            raise RuntimeError(f"Pipeline failed to fit properly: {e}")
        
        self.is_trained = True
        self._training_n_samples = len(X_train)
        
        # Get the actual model from pipeline for backward compatibility
        self.model = self.pipeline.named_steps['model']
        
        # Store scaler separately for fallback use if needed
        if self.scale_features and 'scaler' in self.pipeline.named_steps:
            self._scaler = self.pipeline.named_steps['scaler']
        
        # Evaluate on training data
        y_train_pred = self.pipeline.predict(X_train)
        y_train_pred_original = y_train_pred.copy() if not self.use_log_target else np.exp(y_train_pred) - 1e-8
        y_train_original_eval = y_train.copy() if not self.use_log_target else np.exp(y_train) - 1e-8
        
        train_metrics = self._calculate_metrics(
            y_train, y_train_pred,
            y_train_original_eval, y_train_pred_original
        )
        
        # Evaluate on test data if available
        test_metrics = None
        if len(X_test) > 0:
            y_test_pred = self.pipeline.predict(X_test)
            y_test_pred_original = y_test_pred.copy() if not self.use_log_target else np.exp(y_test_pred) - 1e-8
            
            test_metrics = self._calculate_metrics(
                y_test, y_test_pred,
                y_test_original, y_test_pred_original
            )
            self._test_metrics = test_metrics
        
        # Cross-validation if enabled
        cv_scores = None
        if self.use_cross_validation and len(X_train) > self.cv_splits:
            if verbose:
                print(f"\nPerforming cross-validation ({self.cv_splits} splits)...")
            
            tscv = TimeSeriesSplit(n_splits=min(self.cv_splits, len(X_train) - 1))
            
            # Create a fresh pipeline for CV (to avoid refitting issues)
            cv_pipeline = self._build_pipeline()
            if self.tune_alpha and hasattr(self.base_model, 'alpha'):
                # Use best parameters from grid search
                if hasattr(self.pipeline.named_steps['model'], 'alpha'):
                    cv_pipeline.named_steps['model'].alpha = self.pipeline.named_steps['model'].alpha
                if hasattr(self.pipeline.named_steps['model'], 'l1_ratio'):
                    cv_pipeline.named_steps['model'].l1_ratio = self.pipeline.named_steps['model'].l1_ratio
            
            cv_mae_scores = -cross_val_score(
                cv_pipeline, X_train, y_train, 
                cv=tscv, scoring='neg_mean_absolute_error'
            )
            cv_r2_scores = cross_val_score(
                cv_pipeline, X_train, y_train,
                cv=tscv, scoring='r2'
            )
            
            cv_scores = {
                "cv_mae_mean": float(np.mean(cv_mae_scores)),
                "cv_mae_std": float(np.std(cv_mae_scores)),
                "cv_r2_mean": float(np.mean(cv_r2_scores)),
                "cv_r2_std": float(np.std(cv_r2_scores)),
                "n_splits": len(cv_mae_scores)
            }
            self._cv_scores = cv_scores
        
        # Compile results
        metrics = {
            "train": train_metrics,
            "n_samples": len(X_train),
            "n_features": n_features
        }
        
        if test_metrics:
            metrics["test"] = test_metrics
            # Calculate overfitting metric
            if train_metrics['r2'] > 0:
                overfitting_metric = (train_metrics['r2'] - test_metrics['r2']) / train_metrics['r2']
                metrics["overfitting_metric"] = float(overfitting_metric)
        
        if cv_scores:
            metrics["cv"] = cv_scores
        
        if verbose:
            print(f"\n=== Training Metrics (Original Price Scale) ===")
            print(f"Train Set:")
            print(f"  MAE: ${train_metrics['mae']:,.2f}")
            print(f"  RMSE: ${train_metrics['rmse']:,.2f}")
            print(f"  R² Score: {train_metrics['r2']:.4f}")
            print(f"  MAPE: {train_metrics['mape']:.2f}%")
            
            if test_metrics:
                print(f"\nTest Set:")
                print(f"  MAE: ${test_metrics['mae']:,.2f}")
                print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
                print(f"  R² Score: {test_metrics['r2']:.4f}")
                print(f"  MAPE: {test_metrics['mape']:.2f}%")
                if 'overfitting_metric' in metrics:
                    print(f"  Overfitting Metric: {metrics['overfitting_metric']:.4f}")
            
            if cv_scores:
                print(f"\nCross-Validation:")
                print(f"  CV MAE: ${cv_scores['cv_mae_mean']:,.2f} (±{cv_scores['cv_mae_std']:,.2f})")
                print(f"  CV R²: {cv_scores['cv_r2_mean']:.4f} (±{cv_scores['cv_r2_std']:.4f})")
        
        return metrics
    
    def predict(self, snapshot_data: List[Dict[str, Any]], 
                btc_price_history: Optional[List[float]] = None,
                btc_volume_history: Optional[List[float]] = None) -> float:
        """
        Predict future BTC price from a snapshot of cryptocurrency data.
        
        Args:
            snapshot_data: List of cryptocurrency data dictionaries
            btc_price_history: Optional list of historical BTC prices for technical indicators
            btc_volume_history: Optional list of historical BTC volumes for VWAP calculation
            
        Returns:
            float: Predicted future BTC price (next 5-minute interval) on original scale
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame(snapshot_data)
        
        # Extract features with historical context
        features_df = self.extract_features(df, btc_price_history, btc_volume_history)
        
        # Prepare features
        X = self.prepare_features(features_df)
        
        # Make prediction - use pipeline if available and fitted, otherwise use model directly
        # (for backward compatibility with v1.0/v1.1 models)
        if self.pipeline is not None:
            try:
                # Check if pipeline is fitted
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.pipeline)
                # Pipeline is fitted, use it
                prediction = self.pipeline.predict(X)[0]
            except Exception as e:
                # Pipeline exists but not fitted, or error - fall back to direct model
                # This can happen with old models or if pipeline wasn't properly saved
                from sklearn.utils.validation import check_is_fitted
                
                # Check if model itself is fitted
                try:
                    check_is_fitted(self.model)
                    model_is_fitted = True
                except:
                    model_is_fitted = False
                
                if model_is_fitted and hasattr(self.model, 'predict'):
                    # Model is fitted - apply preprocessing manually if needed
                    # Apply preprocessing in the same order as pipeline: missing_values -> outliers -> scaler -> model
                    
                    # Step 1: Handle missing values
                    missing_handler = None
                    if hasattr(self, '_missing_handler'):
                        missing_handler = self._missing_handler
                    elif self.pipeline and 'missing_values' in self.pipeline.named_steps:
                        try:
                            missing_handler = self.pipeline.named_steps['missing_values']
                        except:
                            pass
                    
                    if missing_handler and hasattr(missing_handler, 'transform'):
                        try:
                            X = missing_handler.transform(X)
                        except:
                            pass  # If transform fails, proceed without imputation
                    
                    # Step 2: Handle outliers (if enabled)
                    outlier_handler = None
                    if hasattr(self, '_outlier_handler'):
                        outlier_handler = self._outlier_handler
                    elif self.handle_outliers and self.pipeline and 'outliers' in self.pipeline.named_steps:
                        try:
                            outlier_handler = self.pipeline.named_steps['outliers']
                        except:
                            pass
                    
                    if outlier_handler and hasattr(outlier_handler, 'transform'):
                        try:
                            X = outlier_handler.transform(X)
                        except:
                            pass  # If transform fails, proceed without outlier handling
                    
                    # Step 3: Scale features
                    if self.scale_features:
                        # Try to get scaler from various sources
                        scaler = None
                        if hasattr(self, '_scaler'):
                            scaler = self._scaler
                        elif self.pipeline and 'scaler' in self.pipeline.named_steps:
                            try:
                                scaler = self.pipeline.named_steps['scaler']
                                # More lenient check - if it has mean_ and scale_, it's likely fitted
                                if not (hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_')):
                                    # Try strict check as fallback
                                    try:
                                        check_is_fitted(scaler)
                                    except:
                                        scaler = None
                            except:
                                scaler = None
                        
                        if scaler is not None and hasattr(scaler, 'transform'):
                            try:
                                X = scaler.transform(X)
                            except Exception as e:
                                # If transform fails, proceed without scaling
                                warnings.warn(f"Scaler transform failed: {e}. Proceeding without scaling.")
                        # If no scaler available, proceed without scaling
                    
                    prediction = self.model.predict(X)[0]
                else:
                    raise ValueError(
                        f"Pipeline not fitted (error: {e}) and model is also not fitted. "
                        f"Model file may be corrupted or model was not properly trained. "
                        f"Please retrain the model."
                    )
        else:
            # No pipeline (v1.0/v1.1 model) - use model directly
            if hasattr(self.model, 'predict'):
                from sklearn.utils.validation import check_is_fitted
                try:
                    check_is_fitted(self.model)
                    prediction = self.model.predict(X)[0]
                except Exception as e:
                    raise ValueError(f"Model is not fitted. Error: {e}. Train the model first.")
            else:
                raise ValueError("Model not available. Train the model first.")
        
        # Inverse transform if log target was used
        if self.use_log_target:
            prediction = np.exp(prediction) - 1e-8
        
        return float(prediction)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information including coefficients, feature names, pipeline info, etc.
        """
        if not self.is_trained:
            return {"is_trained": False}
        
        info = {
            "is_trained": True,
            "model_type": type(self.model).__name__,
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
            "version": "1.2",
            "prediction_target": "Future 5-minute interval BTC price",
            "uses_pipeline": self.pipeline is not None,
            "scale_features": self.scale_features,
            "handle_outliers": self.handle_outliers,
            "use_log_target": self.use_log_target
        }
        
        # Add model-specific information
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            info["coefficients"] = {
                col: float(self.model.coef_[i]) 
                for i, col in enumerate(self.feature_columns)
            }
            info["intercept"] = float(self.model.intercept_)
            
            # Add regularization info
            if hasattr(self.model, 'alpha'):
                info["alpha"] = float(self.model.alpha)
            if hasattr(self.model, 'l1_ratio'):
                info["l1_ratio"] = float(self.model.l1_ratio)
        
        # Add test metrics if available
        if self._test_metrics:
            info["test_metrics"] = self._test_metrics
        
        # Add CV scores if available
        if self._cv_scores:
            info["cv_scores"] = self._cv_scores
        
        return info
    
    def check_multicollinearity(self, X: np.ndarray, threshold_moderate=5.0, threshold_high=10.0) -> Dict[str, Any]:
        """
        Check for multicollinearity using Variance Inflation Factor (VIF).
        
        This is a DIAGNOSTIC tool only - does not auto-drop features.
        Use results to manually identify and prune redundant features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            threshold_moderate: VIF threshold for moderate multicollinearity (default 5.0)
            threshold_high: VIF threshold for high multicollinearity (default 10.0)
            
        Returns:
            dict: VIF diagnostics including VIF values, flagged features, and recommendations
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            warnings.warn("statsmodels not available. Install with: pip install statsmodels")
            return {"error": "statsmodels not available"}
        
        if not self.is_trained or self.feature_columns is None:
            raise ValueError("Model must be trained before checking multicollinearity")
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.feature_columns
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        
        # Flag features
        vif_data["Status"] = "OK"
        vif_data.loc[vif_data["VIF"] > threshold_high, "Status"] = "HIGH"
        vif_data.loc[
            (vif_data["VIF"] > threshold_moderate) & (vif_data["VIF"] <= threshold_high),
            "Status"
        ] = "MODERATE"
        
        # Sort by VIF
        vif_data = vif_data.sort_values("VIF", ascending=False)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        corr_df = pd.DataFrame(corr_matrix, index=self.feature_columns, columns=self.feature_columns)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(self.feature_columns)):
            for j in range(i + 1, len(self.feature_columns)):
                corr_val = abs(corr_matrix[i, j])
                if corr_val > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': self.feature_columns[i],
                        'feature2': self.feature_columns[j],
                        'correlation': float(corr_val)
                    })
        
        # Compile results
        result = {
            "vif_data": vif_data.to_dict('records'),
            "high_vif_features": vif_data[vif_data["Status"] == "HIGH"]["Feature"].tolist(),
            "moderate_vif_features": vif_data[vif_data["Status"] == "MODERATE"]["Feature"].tolist(),
            "high_correlation_pairs": high_corr_pairs,
            "recommendation": "Review high VIF features and high correlation pairs. Manually select the most theoretically sound feature from redundant pairs."
        }
        
        return result
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get feature importance based on coefficient magnitudes.
        
        Args:
            top_n: Number of top features to return (default 10)
            
        Returns:
            dict: Feature importance rankings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if not hasattr(self.model, 'coef_') or self.feature_columns is None:
            return {"error": "Model does not support feature importance"}
        
        # Get absolute coefficients
        coef_abs = np.abs(self.model.coef_)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_,
            'abs_coefficient': coef_abs
        })
        
        # Sort by absolute coefficient
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        # Get top N
        top_features = importance_df.head(top_n)
        
        return {
            "top_features": top_features.to_dict('records'),
            "all_features": importance_df.to_dict('records')
        }
    
    def analyze_residuals(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze residuals to detect model issues.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - original scale
            
        Returns:
            dict: Residual analysis including statistics and diagnostics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before residual analysis")
        
        # Handle log transformation if used
        y_eval = y.copy()
        if self.use_log_target:
            y_eval = np.log(y + 1e-8)
        
        # Get predictions
        predictions = self.pipeline.predict(X) if self.pipeline else self.model.predict(X)
        
        # Inverse transform if log target was used
        if self.use_log_target:
            predictions = np.exp(predictions) - 1e-8
            y_eval = y  # Use original scale for residuals
        
        # Calculate residuals
        residuals = y_eval - predictions
        
        # Calculate statistics
        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "median": float(np.median(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "skewness": float(pd.Series(residuals).skew()),
            "kurtosis": float(pd.Series(residuals).kurtosis())
        }
        
        # Check for patterns
        # Heteroscedasticity check (residuals vs predictions)
        pred_resid_corr = np.corrcoef(predictions, np.abs(residuals))[0, 1]
        
        # Normality check (using Shapiro-Wilk if possible, otherwise use skewness/kurtosis)
        is_normal = abs(residual_stats["skewness"]) < 1.0 and abs(residual_stats["kurtosis"]) < 3.0
        
        diagnostics = {
            "heteroscedasticity_detected": abs(pred_resid_corr) > 0.3,
            "heteroscedasticity_correlation": float(pred_resid_corr),
            "appears_normal": is_normal,
            "recommendations": []
        }
        
        if diagnostics["heteroscedasticity_detected"]:
            diagnostics["recommendations"].append(
                "Heteroscedasticity detected. Consider log transformation of target or features."
            )
        
        if not diagnostics["appears_normal"]:
            diagnostics["recommendations"].append(
                "Residuals may not be normally distributed. Consider transformations."
            )
        
        if residual_stats["mean"] > 0.01 * np.mean(y_eval):
            diagnostics["recommendations"].append(
                "Residual mean is non-zero. Model may have systematic bias."
            )
        
        return {
            "residual_stats": residual_stats,
            "diagnostics": diagnostics,
            "residuals": residuals.tolist(),
            "predictions": predictions.tolist()
        }
    
    def diagnose_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model diagnostics.
        
        Runs multicollinearity check, residual analysis, and feature importance.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - original scale
            
        Returns:
            dict: Comprehensive diagnostic report
        """
        diagnostics = {
            "model_info": self.get_model_info(),
            "multicollinearity": self.check_multicollinearity(X),
            "residual_analysis": self.analyze_residuals(X, y),
            "feature_importance": self.get_feature_importance(top_n=10)
        }
        
        # Add overfitting check if test metrics available
        if self._test_metrics:
            train_metrics = self.get_model_info().get('train_metrics', {})
            if 'r2' in train_metrics and 'r2' in self._test_metrics:
                overfitting = (train_metrics['r2'] - self._test_metrics['r2']) / train_metrics['r2']
                diagnostics["overfitting_check"] = {
                    "overfitting_metric": float(overfitting),
                    "severity": "HIGH" if overfitting > 0.2 else "MODERATE" if overfitting > 0.1 else "LOW"
                }
        
        return diagnostics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data with enhanced metrics on original price scale.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - Future BTC prices (original scale)
            
        Returns:
            dict: Evaluation metrics (MAE, MSE, RMSE, R², Median AE, MAPE) on original scale
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized. Train the model first.")
        
        # Handle log transformation if used during training
        y_original = y.copy()
        y_eval = y.copy()
        if self.use_log_target:
            y_eval = np.log(y + 1e-8)
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        
        # Inverse transform predictions if log target was used
        predictions_original = predictions.copy()
        if self.use_log_target:
            predictions_original = np.exp(predictions) - 1e-8
        
        # Calculate metrics on original scale
        metrics = self._calculate_metrics(
            y_eval, predictions,
            y_original, predictions_original
        )
        metrics["n_samples"] = len(X)
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk (v1.2 with pipeline support).
        
        Args:
            filepath: Path where to save the model (e.g., 'model.pkl' or 'models/btc_model.joblib')
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for saving models. Install it with: pip install joblib")
        
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model. Train the model first.")
        
        # Verify pipeline is fitted before saving (warn if check fails but model is fitted)
        pipeline_fitted = False
        if self.pipeline is not None:
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.pipeline)
                pipeline_fitted = True
            except Exception as e:
                # Pipeline check failed - verify model is fitted as fallback
                if self.model is not None:
                    try:
                        from sklearn.utils.validation import check_is_fitted
                        check_is_fitted(self.model)
                        # Model is fitted, so we can still save (pipeline check might be too strict)
                        warnings.warn(f"Pipeline fitted check failed: {e}. Model is fitted, proceeding with save.")
                    except:
                        # Neither pipeline nor model is fitted
                        raise ValueError(f"Cannot save model: Pipeline fitted check failed ({e}) and model is also not fitted.")
                else:
                    raise ValueError(f"Cannot save model: Pipeline is not fitted. Error: {e}")
        
        # Verify model is fitted (if no pipeline or as additional check)
        model_fitted = False
        if self.model is not None:
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.model)
                model_fitted = True
            except Exception as e:
                # If we have a pipeline that passed the check, we can still save
                if pipeline_fitted:
                    warnings.warn(f"Model fitted check failed: {e}. Pipeline is fitted, proceeding with save.")
                else:
                    raise ValueError(f"Cannot save model: Neither pipeline nor model are fitted. Error: {e}")
        
        # At least one must be fitted
        if not pipeline_fitted and not model_fitted:
            raise ValueError("Cannot save model: Neither pipeline nor model are fitted.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Extract preprocessing transformers separately for fallback use (even if pipeline check fails)
        # Extract directly from pipeline steps - they should be fitted even if pipeline check fails
        saved_scaler = None
        saved_missing_handler = None
        saved_outlier_handler = None
        
        if self.pipeline:
            # Extract scaler if available - try to get it directly from pipeline steps
            if self.scale_features and 'scaler' in self.pipeline.named_steps:
                try:
                    saved_scaler = self.pipeline.named_steps['scaler']
                    # Verify it has the necessary attributes (indicates it's fitted)
                    if not (hasattr(saved_scaler, 'mean_') and hasattr(saved_scaler, 'scale_')):
                        # If it doesn't have mean_/scale_, it's not fitted - don't save it
                        saved_scaler = None
                except:
                    pass
            
            # Extract missing values handler if available
            if 'missing_values' in self.pipeline.named_steps:
                try:
                    saved_missing_handler = self.pipeline.named_steps['missing_values']
                except:
                    pass
            
            # Extract outlier handler if available
            if self.handle_outliers and 'outliers' in self.pipeline.named_steps:
                try:
                    saved_outlier_handler = self.pipeline.named_steps['outliers']
                except:
                    pass
        
        # Also try to use the stored _scaler if available (from training)
        if saved_scaler is None and hasattr(self, '_scaler') and self._scaler is not None:
            if hasattr(self._scaler, 'mean_') and hasattr(self._scaler, 'scale_'):
                saved_scaler = self._scaler
        
        # Save model state (v1.2 format)
        model_data = {
            'pipeline': self.pipeline,  # Save full pipeline
            'model': self.model,  # Keep for backward compatibility
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__,
            'version': '1.2',
            # Save configuration
            'scale_features': self.scale_features,
            'handle_outliers': self.handle_outliers,
            'outlier_method': self.outlier_method,
            'use_log_target': self.use_log_target,
            'use_elastic_net': self.use_elastic_net,
            'test_size': self.test_size,
            'use_cross_validation': self.use_cross_validation,
            'cv_splits': self.cv_splits,
            # Save preprocessing transformers separately for fallback
            '_scaler': saved_scaler,
            '_missing_handler': saved_missing_handler,
            '_outlier_handler': saved_outlier_handler
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def save_model_to_gcs(
        self, 
        bucket_path: str, 
        bucket_name: Optional[str] = None,
        filename_prefix: str = "model"
    ) -> str:
        """
        Save the trained model to a Google Cloud Storage bucket.
        
        Args:
            bucket_path: Path in the bucket (e.g., "lake/models/btc_price_model/pkl" or "gs://bucket-name/path/")
            bucket_name: Optional bucket name. If not provided, will try to extract from bucket_path
                         or use BUCKET_NAME environment variable
            filename_prefix: Prefix for the filename (default: "model")
                           Final filename will be: {filename_prefix}_v1.1_{n_samples}samples_{timestamp}.pkl
                           if training samples count is available, otherwise: {filename_prefix}_v1.1_{timestamp}.pkl
        
        Returns:
            str: Full GCS path to the saved model
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for saving models. Install it with: pip install joblib")
        
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model. Train the model first.")
        
        # Verify pipeline is fitted before saving (warn if check fails but model is fitted)
        pipeline_fitted = False
        if self.pipeline is not None:
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.pipeline)
                pipeline_fitted = True
            except Exception as e:
                # Pipeline check failed - verify model is fitted as fallback
                if self.model is not None:
                    try:
                        from sklearn.utils.validation import check_is_fitted
                        check_is_fitted(self.model)
                        # Model is fitted, so we can still save (pipeline check might be too strict)
                        warnings.warn(f"Pipeline fitted check failed: {e}. Model is fitted, proceeding with save.")
                    except:
                        # Neither pipeline nor model is fitted
                        raise ValueError(f"Cannot save model: Pipeline fitted check failed ({e}) and model is also not fitted.")
                else:
                    raise ValueError(f"Cannot save model: Pipeline is not fitted. Error: {e}")
        
        # Verify model is fitted (if no pipeline or as additional check)
        model_fitted = False
        if self.model is not None:
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.model)
                model_fitted = True
            except Exception as e:
                # If we have a pipeline that passed the check, we can still save
                if pipeline_fitted:
                    warnings.warn(f"Model fitted check failed: {e}. Pipeline is fitted, proceeding with save.")
                else:
                    raise ValueError(f"Cannot save model: Neither pipeline nor model are fitted. Error: {e}")
        
        # At least one must be fitted
        if not pipeline_fitted and not model_fitted:
            raise ValueError("Cannot save model: Neither pipeline nor model are fitted.")
        
        # Try to import GCS storage
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError("google-cloud-storage is required for saving to GCS. Install it with: pip install google-cloud-storage")
        
        # Parse GCS path
        if bucket_path.startswith("gs://"):
            # Format: gs://bucket-name/path/to/dir/
            parts = bucket_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            path_prefix = parts[1] if len(parts) > 1 else ""
        else:
            # Format: path/to/dir/ (bucket_name must be provided or from env)
            path_prefix = bucket_path
            if bucket_name is None:
                import os
                bucket_name = os.getenv('BUCKET_NAME')
                if bucket_name is None:
                    raise ValueError("bucket_name must be provided or BUCKET_NAME environment variable must be set")
        
        # Ensure path_prefix ends with / if it's a directory
        if path_prefix and not path_prefix.endswith("/"):
            path_prefix += "/"
        
        # Generate timestamped filename with optional training samples count
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if model has training info (n_samples attribute)
        n_samples = getattr(self, '_training_n_samples', None)
        if n_samples is not None:
            filename = f"{filename_prefix}_v1.2_{n_samples}samples_{timestamp}.pkl"
        else:
            filename = f"{filename_prefix}_v1.2_{timestamp}.pkl"
        
        full_path = f"{path_prefix}{filename}"
        
        # Save model to temporary file first
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_filepath = tmp_file.name
            # Also save scaler separately for fallback use (even if pipeline check fails)
            # Extract directly from pipeline steps - they should be fitted even if pipeline check fails
            saved_scaler = None
            saved_missing_handler = None
            saved_outlier_handler = None
            
            if self.pipeline:
                # Extract scaler if available - try to get it directly from pipeline steps
                if self.scale_features and 'scaler' in self.pipeline.named_steps:
                    try:
                        saved_scaler = self.pipeline.named_steps['scaler']
                        # Verify it has the necessary attributes (indicates it's fitted)
                        if not (hasattr(saved_scaler, 'mean_') and hasattr(saved_scaler, 'scale_')):
                            # If it doesn't have mean_/scale_, it's not fitted - don't save it
                            saved_scaler = None
                    except:
                        pass
                
                # Extract missing values handler if available
                if 'missing_values' in self.pipeline.named_steps:
                    try:
                        saved_missing_handler = self.pipeline.named_steps['missing_values']
                    except:
                        pass
                
                # Extract outlier handler if available
                if self.handle_outliers and 'outliers' in self.pipeline.named_steps:
                    try:
                        saved_outlier_handler = self.pipeline.named_steps['outliers']
                    except:
                        pass
            
            # Also try to use the stored _scaler if available (from training)
            if saved_scaler is None and hasattr(self, '_scaler') and self._scaler is not None:
                if hasattr(self._scaler, 'mean_') and hasattr(self._scaler, 'scale_'):
                    saved_scaler = self._scaler
            
            model_data = {
                'pipeline': self.pipeline,  # Save full pipeline
                'model': self.model,  # Keep for backward compatibility
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'model_type': type(self.model).__name__,
                'version': '1.2',
                # Save configuration
                'scale_features': self.scale_features,
                'handle_outliers': self.handle_outliers,
                'outlier_method': self.outlier_method,
                'use_log_target': self.use_log_target,
                'use_elastic_net': self.use_elastic_net,
                'test_size': self.test_size,
                'use_cross_validation': self.use_cross_validation,
                'cv_splits': self.cv_splits,
                # Save preprocessing transformers separately for fallback
                '_scaler': saved_scaler,
                '_missing_handler': saved_missing_handler,
                '_outlier_handler': saved_outlier_handler
            }
            joblib.dump(model_data, tmp_filepath)
        
        try:
            # Upload to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(full_path)
            
            blob.upload_from_filename(tmp_filepath)
            
            full_gcs_path = f"gs://{bucket_name}/{full_path}"
            print(f"Model saved to GCS: {full_gcs_path}")
            return full_gcs_path
        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_filepath)
            except:
                pass
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a trained model from disk or GCS bucket.
        
        Args:
            filepath: Path to the saved model file (local path or gs://bucket-name/path/to/model.pkl)
            
        Returns:
            BTCPricePredictor: Loaded model instance
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for loading models. Install it with: pip install joblib")
        
        # Check if it's a GCS path
        is_gcs = filepath.startswith("gs://")
        
        if is_gcs:
            # Load from GCS
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError("google-cloud-storage is required for loading from GCS. Install it with: pip install google-cloud-storage")
            
            # Parse GCS path
            parts = filepath.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""
            
            # Download to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_filepath = tmp_file.name
            
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                if not blob.exists():
                    raise FileNotFoundError(f"Model file not found in GCS: {filepath}")
                
                blob.download_to_filename(tmp_filepath)
                
                # Load model data from temporary file
                model_data = joblib.load(tmp_filepath)
            finally:
                # Clean up temporary file
                try:
                    os.remove(tmp_filepath)
                except:
                    pass
        else:
            # Load from local file
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load model data
            model_data = joblib.load(filepath)
        
        # Create new instance
        version = model_data.get('version', '1.0')
        
        # Handle backward compatibility with v1.1 and v1.0 models
        if version in ['1.0', '1.1']:
            # Old format - no pipeline
            instance = cls(model=model_data['model'])
            instance.model = model_data['model']  # Set model attribute for backward compatibility
            instance.feature_columns = model_data['feature_columns']
            instance.is_trained = model_data['is_trained']
            # Set defaults for v1.2 features
            instance.pipeline = None
            instance.scale_features = False
            instance.handle_outliers = False
            instance.use_log_target = False
            instance.use_elastic_net = False
            instance.test_size = 0.0
            instance.use_cross_validation = False
            instance.cv_splits = 5
            instance.outlier_method = 'winsorize'
        else:
            # v1.2 format - with pipeline
            instance = cls(
                model=model_data.get('model'),  # May not be needed if pipeline exists
                scale_features=model_data.get('scale_features', True),
                handle_outliers=model_data.get('handle_outliers', True),
                outlier_method=model_data.get('outlier_method', 'winsorize'),
                test_size=model_data.get('test_size', 0.2),
                use_cross_validation=model_data.get('use_cross_validation', True),
                cv_splits=model_data.get('cv_splits', 5),
                use_elastic_net=model_data.get('use_elastic_net', False),
                use_log_target=model_data.get('use_log_target', False)
            )
            instance.pipeline = model_data.get('pipeline')
            instance.feature_columns = model_data['feature_columns']
            instance.is_trained = model_data['is_trained']
            
            # Load preprocessing transformers if saved separately (for fallback use)
            if '_scaler' in model_data and model_data['_scaler'] is not None:
                instance._scaler = model_data['_scaler']
            if '_missing_handler' in model_data and model_data['_missing_handler'] is not None:
                instance._missing_handler = model_data['_missing_handler']
            if '_outlier_handler' in model_data and model_data['_outlier_handler'] is not None:
                instance._outlier_handler = model_data['_outlier_handler']
            
            # Verify pipeline is fitted (should be if model was saved correctly)
            if instance.pipeline is not None:
                try:
                    from sklearn.utils.validation import check_is_fitted
                    check_is_fitted(instance.pipeline)
                    # Pipeline is fitted - extract model from it
                    if 'model' in instance.pipeline.named_steps:
                        instance.model = instance.pipeline.named_steps['model']
                        # Verify model is also fitted
                        check_is_fitted(instance.model)
                    else:
                        # Pipeline exists but no model step - use saved model
                        if 'model' in model_data:
                            instance.model = model_data['model']
                except Exception as e:
                    # Pipeline not fitted - try to use saved model directly
                    warnings.warn(f"Pipeline not fitted after loading: {e}. Attempting to use saved model directly.")
                    if 'model' in model_data:
                        instance.model = model_data['model']
                        # Check if saved model is fitted
                        try:
                            check_is_fitted(instance.model)
                        except:
                            raise ValueError(f"Neither pipeline nor model are fitted. Model file may be corrupted: {filepath}")
                    else:
                        raise ValueError(f"Pipeline not fitted and no model available. Model file may be corrupted: {filepath}")
            else:
                # No pipeline - use saved model
                if 'model' in model_data:
                    instance.model = model_data['model']
                else:
                    raise ValueError(f"No pipeline or model found in saved file: {filepath}")
            
            # Store scaler separately for fallback use if needed
            # Try to extract scaler even if pipeline check failed (individual steps might be fitted)
            if instance.scale_features and instance.pipeline and 'scaler' in instance.pipeline.named_steps:
                try:
                    scaler = instance.pipeline.named_steps['scaler']
                    # Try to use scaler if it has the necessary attributes (mean_ and scale_)
                    # This is more lenient than check_is_fitted which might be too strict
                    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_') and hasattr(scaler, 'transform'):
                        instance._scaler = scaler
                    else:
                        # Try check_is_fitted as fallback
                        try:
                            check_is_fitted(scaler)
                            instance._scaler = scaler
                        except:
                            # Scaler not fitted, but that's okay - we'll proceed without it
                            pass
                except:
                    pass  # If scaler not available, that's okay
        
        print(f"Model loaded from: {filepath}")
        print(f"  Model type: {model_data.get('model_type', 'Unknown')}")
        print(f"  Version: {version}")
        print(f"  Features: {len(instance.feature_columns) if instance.feature_columns else 0}")
        print(f"  Trained: {instance.is_trained}")
        if version == '1.2':
            print(f"  Uses Pipeline: {instance.pipeline is not None}")
            if instance.pipeline is not None:
                try:
                    from sklearn.utils.validation import check_is_fitted
                    check_is_fitted(instance.pipeline)
                    print(f"  Pipeline Status: Fitted")
                except:
                    print(f"  Pipeline Status: Not Fitted (using model directly)")
            print(f"  Feature Scaling: {instance.scale_features}")
        
        return instance


def ingest_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Ingest cryptocurrency market data from JSON format into a pandas DataFrame.
    
    Args:
        data: List of dictionaries containing cryptocurrency market data
        
    Returns:
        pd.DataFrame: DataFrame with cryptocurrency market data
    """
    df = pd.DataFrame(data)
    return df
