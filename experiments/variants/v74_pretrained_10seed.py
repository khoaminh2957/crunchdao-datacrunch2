"""v74: PRETRAINED 10-seed XGB ensemble (trained locally, no cloud re-train).

The model.pkl in resources/ contains (models_list, feature_cols, target_col).
train() is a NO-OP — Force first train=No on trigger ensures cloud loads our model directly.
"""
import pandas as pd, numpy as np, pickle, os


def train(X_train, y_train, model_directory_path):
    """No-op: model already pretrained and shipped in resources/."""
    print("[v74/train] using PRE-TRAINED 10-seed ensemble (no retrain)")
    if not os.path.exists(f"{model_directory_path}/model.pkl"):
        raise FileNotFoundError("Expected pre-trained model.pkl in resources/")


def infer(X_test, model_directory_path):
    with open(f"{model_directory_path}/model.pkl", "rb") as f:
        models, feature_cols, _ = pickle.load(f)
    print(f"[v74/infer] loaded {len(models)}-model ensemble, predicting on {len(X_test)} rows")
    preds = np.mean([m.predict(X_test[feature_cols]) for m in models], axis=0)
    id_col = "id" if "id" in X_test.columns else "Id"
    moon_col = "moon" if "moon" in X_test.columns else "Moon"
    return pd.DataFrame({id_col: X_test[id_col].values, moon_col: X_test[moon_col].values, "prediction": preds})
