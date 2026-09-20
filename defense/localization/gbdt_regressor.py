from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from defense.utils.logger import log

class GBDTRegressor:
    """Multi-target GBDT regressor for drone geo-localization."""

    TARGETS = ["lat", "lon", "alt"]

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = Path(models_dir)
        self.models = {}       # {"lat": GBR, "lon": GBR, "alt": GBR}
        self.scaler = None     # StandardScaler fitted on training features

    def train(self, features: np.ndarray, targets: dict,
              n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 5,
              min_samples_split: int = 5, min_samples_leaf: int = 2, subsample: float = 0.9):
        """
        Train 3 GBDT regressors.

        Args:
            features: np.ndarray of shape (N, feature_dim)
            targets: dict {"lat": np.ndarray, "lon": np.ndarray, "alt": np.ndarray}
        """
        log("GBDT", f"Training on {features.shape[0]} samples, {features.shape[1]} features")

        # 1. Fit StandardScaler on features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # 2. Train one GBR per target
        for target_name in self.TARGETS:
            log("GBDT", f"  Training {target_name} regressor (n_estimators={n_estimators})...")
            gbr = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=42,
            )
            gbr.fit(X_scaled, targets[target_name])
            self.models[target_name] = gbr
            r2 = gbr.score(X_scaled, targets[target_name])
            log("GBDT", f"  ✅ {target_name} regressor trained (R² on train: {r2:.4f})")

        log("GBDT", "✅ All 3 regressors trained successfully")

    def save(self):
        """Save models and scaler to models_dir/"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            path = self.models_dir / f"{name}.bin"
            joblib.dump(model, path)
            log("GBDT", f"  Saved: {path}")
        scaler_path = self.models_dir / "scaler.bin"
        joblib.dump(self.scaler, scaler_path)
        log("GBDT", f"  Saved: {scaler_path}")
        log("GBDT", f"✅ All models saved to {self.models_dir}")

    def load(self):
        """Load models and scaler. Raises FileNotFoundError with helpful message if missing."""
        required = [f"{t}.bin" for t in self.TARGETS] + ["scaler.bin"]
        for fname in required:
            path = self.models_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"GBDT model file not found: {path}\n"
                    f"To train these models, run:\n"
                    f"  uv run python scripts/train_gbdt.py --data <training_data_dir> --out {self.models_dir}"
                )
        for name in self.TARGETS:
            self.models[name] = joblib.load(self.models_dir / f"{name}.bin")
        self.scaler = joblib.load(self.models_dir / "scaler.bin")
        log("GBDT", f"✅ Loaded models from {self.models_dir}: {', '.join(f'{t}.bin' for t in self.TARGETS)}, scaler.bin")

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict lat, lon, alt from feature vector(s).

        Args:
            features: np.ndarray of shape (feature_dim,) or (N, feature_dim)

        Returns:
            dict {"lat": float/array, "lon": float/array, "alt": float/array}
        """
        is_1d = (features.ndim == 1)
        if is_1d:
            features = features.reshape(1, -1)
        X_scaled = self.scaler.transform(features)
        preds = {name: self.models[name].predict(X_scaled) for name in self.TARGETS}
        if is_1d:
            return {name: float(preds[name][0]) for name in self.TARGETS}
        return preds
