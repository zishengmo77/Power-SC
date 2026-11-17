import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

plt.rcParams['figure.dpi'] = 300  # Set display DPI

# =========================
# 1) Load data
# =========================
file_path = 'wqwdata.xlsx'
data = pd.read_excel(file_path)

data_num = data.select_dtypes(include=[np.number]).copy()
data_num = data_num.replace([np.inf, -np.inf], np.nan)
data_num = data_num.fillna(data_num.median(numeric_only=True))

target_col = 'SC'
if target_col not in data_num.columns:
    raise ValueError(f"Target column {target_col} not found. Available columns: {data_num.columns.tolist()}")

X = data_num.drop(columns=[target_col])
y = data_num[target_col]

print("X shape:", X.shape, "y shape:", y.shape)
print("Features:", X.columns.tolist())


# =========================
# 2) Predicted vs. experimental plot with 95% prediction interval
# =========================
def plot_pred_vs_actual(y_train, y_train_pred, y_test, y_test_pred, model_name="SC", mse_cv=None):
    fig, ax_scatter = plt.subplots(figsize=(11, 8))

    ax_scatter.scatter(y_train, y_train_pred, color='gray', alpha=0.6, label='Train data', s=100)
    ax_scatter.scatter(y_test, y_test_pred, color='orange', alpha=0.6, label='Test data', s=100)

    all_actual = np.concatenate([y_train.values, y_test.values])
    all_pred = np.concatenate([y_train_pred, y_test_pred])

    errors = all_pred - all_actual
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    mean_actual = np.mean(all_actual)
    std_dev_min, std_dev_max = std_error * 0.5, std_error * 2.5
    x_range = np.linspace(all_actual.min(), all_actual.max(), 100)
    std_devs = std_dev_min + (std_dev_max - std_dev_min) * np.abs(x_range - mean_actual) / (
                all_actual.max() - all_actual.min())
    std_devs = np.clip(std_devs, std_dev_min, std_dev_max)

    ax_scatter.fill_between(x_range,
                            x_range + mean_error - 2 * std_devs,
                            x_range + mean_error + 2 * std_devs,
                            color='grey', alpha=0.2, label='95% Prediction Interval')

    y_min = min(min(y_train), min(y_test))
    y_max = max(max(y_train), max(y_test))
    ax_scatter.plot([y_min, y_max], [y_min, y_max], color='black', linestyle='--', label='y = x')

    ax_scatter.set_xlabel('Experimental values', fontsize=24, fontweight='bold')
    ax_scatter.set_ylabel('Predictive values', fontsize=24, fontweight='bold')
    ax_scatter.tick_params(axis='both', which='major', labelsize=18)
    ax_scatter.legend(fontsize=14)
    ax_scatter.grid(True, alpha=0.5)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    print(f'\n[{model_name}] Train $R^2$: {r2_train:.2f}, RMSE: {rmse_train:.4f}, MSE: {mse_train:.4f}')
    print(f'[{model_name}] Test $R^2$: {r2_test:.2f}, RMSE: {rmse_test:.4f}, MSE: {mse_test:.4f}')
    if mse_cv is not None:
        print(f'[{model_name}] Cross-validation MSE: {mse_cv:.4f}')

    ax_scatter.text(0.05, 0.95, f'$R^2_{{Train}}$: {r2_train:.2f}',
                    transform=ax_scatter.transAxes, fontsize=16, va='top')
    ax_scatter.text(0.05, 0.88, f'RMSE$_{{Train}}$: {rmse_train:.2f}',
                    transform=ax_scatter.transAxes, fontsize=16, va='top')
    ax_scatter.text(0.05, 0.81, f'$R^2_{{Test}}$: {r2_test:.2f}',
                    transform=ax_scatter.transAxes, fontsize=16, va='top')
    ax_scatter.text(0.05, 0.74, f'RMSE$_{{Test}}$: {rmse_test:.2f}',
                    transform=ax_scatter.transAxes, fontsize=16, va='top')

    plt.tight_layout()
    plt.savefig(f"R2_{model_name}", dpi=300, bbox_inches='tight')
    plt.show()


# =========================
# 3) Train, evaluate and plot
# =========================
def train_eval_plot_for_SC(X, y, model_name="SC"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.5,
        random_state=42
    )
    xgb.fit(X_train, y_train)

    y_train_pred = xgb.predict(X_train)
    y_test_pred = xgb.predict(X_test)

    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(xgb, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_cv = -cv_scores.mean()

    plot_pred_vs_actual(y_train, y_train_pred, y_test, y_test_pred, model_name=model_name, mse_cv=mse_cv)

    return xgb


print("Training XGBoost and generating visualization for SC ...")
model_SC = train_eval_plot_for_SC(X, y, model_name="SC")
