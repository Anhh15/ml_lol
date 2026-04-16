import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

BEST_MODEL = 'xgb'

BEST_PARAMS = {
    # XGBoost — best params tu t1_modeling_xgb.ipynb (CV Acc: 52.49%)
    'learning_rate': 0.05,
    'max_depth':     7,
    'n_estimators':  300,
    'subsample':     0.8,
    'eval_metric':   'logloss',
    'random_state':  42,
    'n_jobs':        -1,
    'verbosity':     0,
}

# ============================================================
# --- Paths ---
ROOT      = Path(__file__).resolve().parents[2]
PROC_DIR  = ROOT / 'data' / 'processed'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH      = PROC_DIR / 't1_train.csv'
MODEL_OUT       = MODEL_DIR / 't1_best_model.pkl'
SCALER_OUT      = MODEL_DIR / 't1_scaler.pkl'

CV_FOLDS        = 5
RANDOM_STATE    = 42


def build_model(name: str, params: dict):
    name = name.lower().strip()

    if name == 'lr':
        from sklearn.linear_model import LogisticRegression
        # Loc parms chi cua LR
        lr_keys = {'C', 'solver', 'max_iter', 'random_state', 'penalty', 'l1_ratio'}
        lr_params = {k: v for k, v in params.items() if k in lr_keys}
        lr_params.setdefault('max_iter', 1000)
        lr_params.setdefault('random_state', RANDOM_STATE)
        return LogisticRegression(**lr_params), True   # True = can scale

    elif name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        rf_params = dict(params)
        rf_params.setdefault('random_state', RANDOM_STATE)
        rf_params.setdefault('n_jobs', -1)
        return RandomForestClassifier(**rf_params), False

    elif name == 'xgb':
        from xgboost import XGBClassifier
        xgb_params = dict(params)
        xgb_params.setdefault('eval_metric', 'logloss')
        xgb_params.setdefault('random_state', RANDOM_STATE)
        xgb_params.setdefault('n_jobs', -1)
        xgb_params.setdefault('verbosity', 0)
        return XGBClassifier(**xgb_params), False

    elif name == 'lgbm':
        from lightgbm import LGBMClassifier
        lgbm_params = dict(params)
        lgbm_params.setdefault('random_state', RANDOM_STATE)
        lgbm_params.setdefault('n_jobs', -1)
        lgbm_params.setdefault('verbose', -1)
        return LGBMClassifier(**lgbm_params), False

    else:
        raise ValueError(f"Model '{name}' khong hop le. Chon: lr | rf | xgb | lgbm")


def train() -> None:
    # --- Load train ---
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=['winner'])
    y = train_df['winner']

    # --- Build model ---
    model, need_scale = build_model(BEST_MODEL, BEST_PARAMS)

    # --- Scaling neu can (chi LR) ---
    if need_scale:
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        scaler = StandardScaler()
        model = Pipeline([('scaler', scaler), ('model', model)])


    # --- Cross-validation tren toan bo tap train ---
    cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)

    # --- Train tren toan bo tap train ---
    model.fit(X, y)
    train_acc = (model.predict(X) == y).mean()

    # --- Luu model ---
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)

train()
