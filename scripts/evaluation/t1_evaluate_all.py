import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive backend, khong can GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# -- Paths ---------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / 'data' / 'processed'
RES_DIR  = ROOT / 'results'
PLOT_DIR = RES_DIR / 'plots'

TEST_PATH  = PROC_DIR / 't1_test.csv'
MODEL_PATH = ROOT / 'models' / 't1_best_model.pkl'

RES_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -- Style ---------------------------------------------------------------------
sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['figure.dpi'] = 130
C1, C2 = '#4C82C2', '#E07B54'


# -- Helpers -------------------------------------------------------------------

def get_model_name(model) -> str:
    name = type(model).__name__
    if name == 'Pipeline':
        name = type(model.named_steps['model']).__name__
    return name


def save_fig(path: Path) -> None:
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, model_name, acc) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['T1 Thang', 'T2 Thang'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Ma tran nham lan — {model_name}\nAccuracy = {acc:.4f}', fontsize=12)
    plt.tight_layout()
    save_fig(PLOT_DIR / 't1_confusion_matrix.png')


def plot_roc_curve(y_test, y_prob, model_name, auc) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=C1, lw=2, label=f'{model_name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color=C1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    save_fig(PLOT_DIR / 't1_roc_curve.png')


def plot_pr_curve(y_test, y_prob, model_name, ap) -> None:
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    baseline = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color=C2, lw=2, label=f'{model_name} (AP={ap:.4f})')
    ax.axhline(baseline, color='gray', linestyle='--', lw=1,
               label=f'Baseline ({baseline:.2f})')
    ax.fill_between(rec, prec, alpha=0.1, color=C2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    plt.tight_layout()
    save_fig(PLOT_DIR / 't1_pr_curve.png')


def plot_feature_importance(model, feature_names, model_name) -> None:
    fi = None
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        inner = model.named_steps['model']
        if hasattr(inner, 'coef_'):
            fi = np.abs(inner.coef_[0])
        elif hasattr(inner, 'feature_importances_'):
            fi = inner.feature_importances_
    elif hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
    elif hasattr(model, 'coef_'):
        fi = np.abs(model.coef_[0])

    if fi is None:
        return

    fi_s = pd.Series(fi, index=feature_names).nlargest(15)
    label = '|Coefficient|' if 'Logistic' in model_name else 'Feature Importance'
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fi_s.index[::-1], fi_s.values[::-1], color=C1)
    ax.set_xlabel(label)
    ax.set_title(f'Top 15 Feature Quan Trong — {model_name}')
    plt.tight_layout()
    save_fig(PLOT_DIR / 't1_feature_importance.png')


def plot_metrics_bar(metrics: dict, model_name: str) -> None:
    keys   = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    values = [metrics[k] for k in keys]
    colors = [C1 if v >= 0.55 else C2 for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(keys, values, color=colors)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Random baseline')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(f'Tong hop Metrics — {model_name} (T1: Draft Only)')
    ax.legend()
    plt.tight_layout()
    save_fig(PLOT_DIR / 't1_metrics_summary.png')


# -- Main ----------------------------------------------------------------------

def evaluate() -> None:
    # Load test
    test_df = pd.read_csv(TEST_PATH)
    X_test  = test_df.drop(columns=['winner'])
    y_test  = test_df['winner']

    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    model_name = get_model_name(model)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        'accuracy':       round(accuracy_score(y_test, y_pred), 4),
        'precision':      round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':         round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':             round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':        round(roc_auc_score(y_test, y_prob), 4),
        'log_loss':       round(log_loss(y_test, y_prob), 4),
        'avg_precision':  round(average_precision_score(y_test, y_prob), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred,
                                target_names=['T1 Thang', 'T2 Thang'])

    # Plots
    plot_confusion_matrix(cm, model_name, metrics['accuracy'])
    plot_roc_curve(y_test, y_prob, model_name, metrics['roc_auc'])
    plot_pr_curve(y_test, y_prob, model_name, metrics['avg_precision'])
    plot_feature_importance(model, list(X_test.columns), model_name)
    plot_metrics_bar(metrics, model_name)

    # Save CSV
    row = {'experiment': 'T1_Draft_Only', 'model': model_name,
           'test_samples': len(y_test)}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(RES_DIR / 't1_metrics_summary.csv', index=False)

    # Save report
    report_path = RES_DIR / 't1_classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('Experiment : T1 — Draft Only\n')
        f.write(f'Model      : {model_name}\n')
        f.write(f'Test set   : {len(y_test)} samples\n')
        f.write('-' * 50 + '\n')
        f.write(cr)
        f.write('-' * 50 + '\n')
        f.write(f'ROC-AUC          : {metrics["roc_auc"]}\n')
        f.write(f'Log Loss         : {metrics["log_loss"]}\n')
        f.write(f'Average Precision: {metrics["avg_precision"]}\n')

evaluate()
