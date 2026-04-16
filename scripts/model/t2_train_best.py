"""
t2_train_best.py
================
Giai đoạn 4 — Train model tốt nhất với dữ liệu T2 (Pre-game + Early Game 15 mins)

Output:
    models/t2_best_model.pkl
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# ============================================================
# CONFIGURATION — (Thay đổi tham số này nếu cần sau khi Tuning)
# ============================================================

BEST_MODEL = 'xgb'     # 'rf' | 'xgb'

BEST_PARAMS = {
    # XGBoost Default
    'learning_rate': 0.05,
    'max_depth':     5,
    'n_estimators':  300,
    'subsample':     0.8,
    
    # Cấu hình tĩnh
    'eval_metric':   'logloss',
    'random_state':  42,
    'n_jobs':        -1,
    'verbosity':     0,
}

# ============================================================

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# --- Paths ---
ROOT      = Path(__file__).resolve().parents[2]
PROC_DIR  = ROOT / 'data' / 'processed'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH      = PROC_DIR / 't2_train.csv'
MODEL_OUT       = MODEL_DIR / 't2_best_model.pkl'

CV_FOLDS        = 5
RANDOM_STATE    = 42


def build_model(name: str, params: dict):
    """Khoi tao model theo ten."""
    name = name.lower().strip()

    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        rf_params = dict(params)
        rf_params.setdefault('random_state', RANDOM_STATE)
        rf_params.setdefault('n_jobs', -1)
        return RandomForestClassifier(**rf_params)

    elif name == 'xgb':
        from xgboost import XGBClassifier
        xgb_params = dict(params)
        xgb_params.setdefault('eval_metric', 'logloss')
        xgb_params.setdefault('random_state', RANDOM_STATE)
        xgb_params.setdefault('n_jobs', -1)
        xgb_params.setdefault('verbosity', 0)
        return XGBClassifier(**xgb_params)

    else:
        raise ValueError(f"Model '{name}' khong hop le. Chon: rf | xgb")


def train() -> None:
    # --- Load train ---
    log.info('Load t2_train.csv ...')
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=['winner'])
    y = train_df['winner']
    log.info(f'  Shape: {X.shape}  |  Target: {y.value_counts().to_dict()}')

    # --- Build model ---
    log.info(f'Model da chon : {BEST_MODEL.upper()}')
    log.info(f'Best params   : {BEST_PARAMS}')
    model = build_model(BEST_MODEL, BEST_PARAMS)

    # --- Cross-validation tren toan bo tap train ---
    log.info(f'Cross-validation (cv={CV_FOLDS}) ...')
    cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    log.info(f'  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
    log.info(f'  Scores ca fold: {[round(s, 4) for s in cv_scores]}')

    # --- Train tren toan bo tap train ---
    log.info('Train tren toan bo tap train ...')
    model.fit(X, y)
    train_acc = (model.predict(X) == y).mean()
    log.info(f'  Training Accuracy: {train_acc:.4f}')

    # --- Luu model ---
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
    log.info(f'Da luu model: {MODEL_OUT}')

    # --- Summary ---
    print('\n-- Summary -------------------------------------------')
    print(f'  Model         : {BEST_MODEL.upper()}')
    print(f'  CV Accuracy   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
    print(f'  Train Accuracy: {train_acc:.4f}')
    print(f'  Model saved   : {MODEL_OUT}')
    print(f'  Features used : {X.shape[1]} cols')
    print('------------------------------------------------------')


if __name__ == '__main__':
    train()
