import numpy as np
from defense.localization.gbdt_regressor import GBDTRegressor

def test_gbdt_train_predict(tmp_path):
    reg = GBDTRegressor(models_dir=str(tmp_path))
    # 20 samples with 20 features
    X = np.random.randn(20, 20)
    targets = {
        "lat": np.linspace(14.0, 14.5, 20),
        "lon": np.linspace(101.0, 101.5, 20),
        "alt": np.linspace(30.0, 50.0, 20),
    }
    reg.train(X, targets, n_estimators=10)
    reg.save()

    # Load back
    reg_loaded = GBDTRegressor(models_dir=str(tmp_path))
    reg_loaded.load()

    # Predict single sample
    sample = X[0]
    preds = reg_loaded.predict(sample)
    assert "lat" in preds and "lon" in preds and "alt" in preds
    assert isinstance(preds["lat"], float)
