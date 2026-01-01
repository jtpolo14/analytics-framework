"""
BTC Price Prediction Model v1

This module contains the model class for predicting Bitcoin prices based on
cryptocurrency market data. The model can be easily swapped or extended with
different algorithms while maintaining the same interface.

NOTE: This is a trivial model implementation for framework demonstration purposes.
It uses simple linear regression (Ridge) with basic feature engineering.
For production use, consider more sophisticated models (e.g., time series models,
ensemble methods, or deep learning approaches).
"""
import pandas as pd
import numpy as np
import time
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Any, Tuple, Optional
import json
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class BTCPricePredictor:
    """
    Model for predicting Bitcoin prices based on cryptocurrency market data.
    
    This class encapsulates the feature extraction, model training, and prediction
    logic. Different model implementations can be swapped by changing the underlying
    sklearn model or implementing custom training logic.
    """
    
    def __init__(self, model=None, use_regularization=True):
        """
        Initialize the BTC price predictor.
        
        Args:
            model: Optional sklearn-compatible model. If None, uses LinearRegression or Ridge.
            use_regularization: If True and model is None, uses Ridge regression to prevent overfitting.
                               If False, uses plain LinearRegression.
        """
        if model is None:
            # Use Ridge regression by default to prevent overfitting
            # Ridge adds L2 regularization which helps when we have more features than samples
            self.model = Ridge(alpha=1.0) if use_regularization else LinearRegression()
        else:
            self.model = model
        self.feature_columns = None
        self.is_trained = False
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from cryptocurrency data for prediction model.
        Creates features that might be useful for predicting BTC price.
        
        Args:
            df: DataFrame with cryptocurrency market data
            
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
            # BTC-specific features
            feature_row['btc_volume'] = btc_data.get('total_volume', 0)
            feature_row['btc_price_change_24h'] = btc_data.get('price_change_24h', 0)
            feature_row['btc_price_change_pct_24h'] = btc_data.get('price_change_percentage_24h', 0)
            feature_row['btc_market_cap'] = btc_data.get('market_cap', 0)
            feature_row['btc_high_24h'] = btc_data.get('high_24h', 0)
            feature_row['btc_low_24h'] = btc_data.get('low_24h', 0)
            feature_row['btc_price_change_pct_7d'] = btc_data.get('price_change_percentage_7d_in_currency', 0)
            feature_row['btc_price_change_pct_30d'] = btc_data.get('price_change_percentage_30d_in_currency', 0)
            feature_row['btc_ath_change_pct'] = btc_data.get('ath_change_percentage', 0)
        
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
        
        Args:
            snapshots: List of snapshots, where each snapshot is a list of 
                      cryptocurrency data dictionaries
            
        Returns:
            tuple: (X, y) where:
                - X: Feature matrix (n_samples, n_features)
                - y: Target vector (n_samples,) - BTC prices
        """
        all_features = []
        all_targets = []
        total_records = 0
        
        for i, snapshot_data in enumerate(snapshots):
            snapshot_start = time.time()
            # Convert to DataFrame
            df = pd.DataFrame(snapshot_data)
            total_records += len(df)
            
            # Extract features
            features_df = self.extract_features(df)
            
            # Get BTC price (target)
            btc_data = df[df['symbol'] == 'btc'].iloc[0] if len(df[df['symbol'] == 'btc']) > 0 else None
            if btc_data is None:
                print(f"Warning: Snapshot {i+1} missing BTC data, skipping...")
                continue
            
            current_btc_price = btc_data['current_price']
            last_updated = btc_data.get('last_updated', 'N/A')
            
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
            all_targets.append(current_btc_price)
            
            snapshot_duration = time.time() - snapshot_start
            print(f"  Snapshot {i+1}: BTC Price = ${current_btc_price:,.2f} | Last Updated: {formatted_time} | ({snapshot_duration:.3f}s)")
        
        if len(all_features) == 0:
            raise ValueError("No valid snapshots found. At least one snapshot with BTC data is required.")
        
        X = np.array(all_features)
        y = np.array(all_targets)
        
        print(f"\nBuilt training dataset: {len(all_features)} samples, {len(self.feature_columns)} features")
        print(f"Total records processed: {total_records:,}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Train the model on provided features and targets.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            verbose: Whether to print training information
            
        Returns:
            dict: Training metrics (MAE, MSE, R²)
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
        
        n_samples, n_features = X.shape
        
        # Warn about potential overfitting
        if n_samples <= n_features:
            if verbose:
                print(f"\n⚠️  WARNING: Overfitting risk detected!")
                print(f"   Training samples ({n_samples}) <= Features ({n_features})")
                print(f"   Model may memorize training data instead of learning patterns.")
                print(f"   Consider: using more data, fewer features, or regularization.")
        
        if verbose:
            print(f"\nTraining model on {n_samples} samples with {n_features} features...")
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        # Store number of training samples for use in model filename
        self._training_n_samples = n_samples
        
        # Evaluate on training data
        train_predictions = self.model.predict(X)
        train_mae = mean_absolute_error(y, train_predictions)
        train_mse = mean_squared_error(y, train_predictions)
        train_r2 = r2_score(y, train_predictions)
        
        metrics = {
            "mae": float(train_mae),
            "mse": float(train_mse),
            "r2": float(train_r2),
            "n_samples": len(X)
        }
        
        if verbose:
            print(f"Training Metrics:")
            print(f"  MAE: ${train_mae:,.2f}")
            print(f"  MSE: ${train_mse:,.2f}")
            print(f"  R² Score: {train_r2:.4f}")
        
        return metrics
    
    def predict(self, snapshot_data: List[Dict[str, Any]]) -> float:
        """
        Predict BTC price from a snapshot of cryptocurrency data.
        
        Args:
            snapshot_data: List of cryptocurrency data dictionaries
            
        Returns:
            float: Predicted BTC price
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame(snapshot_data)
        
        # Extract features
        features_df = self.extract_features(df)
        
        # Prepare features
        X = self.prepare_features(features_df)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return float(prediction)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information including coefficients, feature names, etc.
        """
        if not self.is_trained:
            return {"is_trained": False}
        
        info = {
            "is_trained": True,
            "model_type": type(self.model).__name__,
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns) if self.feature_columns else 0
        }
        
        # Add model-specific information
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            info["coefficients"] = {
                col: float(self.model.coef_[i]) 
                for i, col in enumerate(self.feature_columns)
            }
            info["intercept"] = float(self.model.intercept_)
        
        return info
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            
        Returns:
            dict: Evaluation metrics (MAE, MSE, R²)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.model.predict(X)
        
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            "mae": float(mae),
            "mse": float(mse),
            "r2": float(r2),
            "n_samples": len(X)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model (e.g., 'model.pkl' or 'models/btc_model.joblib')
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for saving models. Install it with: pip install joblib")
        
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model state
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__
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
                           Final filename will be: {filename_prefix}_{n_samples}samples_{timestamp}.pkl
                           if training samples count is available, otherwise: {filename_prefix}_{timestamp}.pkl
        
        Returns:
            str: Full GCS path to the saved model
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for saving models. Install it with: pip install joblib")
        
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model. Train the model first.")
        
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
            filename = f"{filename_prefix}_{n_samples}samples_{timestamp}.pkl"
        else:
            filename = f"{filename_prefix}_{timestamp}.pkl"
        
        full_path = f"{path_prefix}{filename}"
        
        # Save model to temporary file first
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_filepath = tmp_file.name
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'model_type': type(self.model).__name__
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
        instance = cls(model=model_data['model'])
        instance.feature_columns = model_data['feature_columns']
        instance.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {filepath}")
        print(f"  Model type: {model_data.get('model_type', 'Unknown')}")
        print(f"  Features: {len(instance.feature_columns) if instance.feature_columns else 0}")
        print(f"  Trained: {instance.is_trained}")
        
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
