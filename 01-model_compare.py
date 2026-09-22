import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
import time

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Arial', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['savefig.dpi'] = 600

# ==========================================
data_path = r"dataset.xlsx"
output_metrics = "multimodel_aug_metrics.xlsx"
output_fig_png = "multimodel_fitting"

# 加速关键参数
n_aug_levels = 10
aug_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
noise_factor = 0.035
test_size = 0.3
random_state = 42
cv_folds = 5
use_random_search = True
random_iter = 12

np.random.seed(random_state)

# ==========================================
FEATURE_COLS = [
    'Plant height', 'Number of main stems',
    'SL_FW', 'SS_FW', 'SR_FW', 'SL_DW', 'SS_DW', 'SR_DW',
    'TL_FW', 'TS_FW', 'TR_FW', 'TL_DW', 'TS_DW', 'TR_DW',
    'EL_FW', 'ES_FW', 'ER_FW', 'ES_DW', 'EL_DW', 'ER_DW',
    'AL_FW', 'AS_FW', 'AR_FW', 'AS_DW', 'AL_DW', 'AR_DW',
    'AN', 'BD', 'OC', 'TN', 'Clay', 'pH', 'sand',
    'S_prec', 'T_prec', 'E_prec', 'A_prec', 'M_prec',
    'S_srad', 'T_srad', 'E_srad', 'A_srad', 'M_srad',
    'S_GDD', 'S-T_GDD', 'S-E_GDD', 'S-A_GDD', 'S-M_GDD',
    'N_rate'
]

# ==========================================
df_raw = pd.read_excel(data_path, sheet_name='Sheet1')
df_raw['env_key'] = df_raw['year'].astype(str) + '_' + df_raw['region'].astype(str)

train_dfs, test_dfs = [], []
for key, group in df_raw.groupby('env_key'):
    if len(group) < 2:
        train_dfs.append(group)
    else:
        tr, te = train_test_split(group, test_size=test_size, random_state=random_state)
        train_dfs.append(tr)
        test_dfs.append(te)

df_train = pd.concat(train_dfs, ignore_index=True)
df_test = pd.concat(test_dfs, ignore_index=True)

target = 'weight'

# =========================================
def preprocess(df, feature_cols, target_col):
    y_ori = df[target_col].values
    y = np.log1p(y_ori)
    feature_cols_used = [col for col in feature_cols if col != 'number']
    X = df[feature_cols_used].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.mean(numeric_only=True))
    return X, y, y_ori

X_train, y_train, y_train_ori = preprocess(df_train, FEATURE_COLS, target)
X_test, y_test, y_test_ori = preprocess(df_test, FEATURE_COLS, target)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()


# ==========================================
def generate_all_augmented_datasets(X, y, numeric_cols, aug_levels, noise_factor):
    X_num = X[numeric_cols].values.astype(float)
    n_samples, n_features = X_num.shape
    cov = np.cov(X_num, rowvar=False)
    noise_cov = (noise_factor ** 2) * cov
    try:
        L = np.linalg.cholesky(noise_cov)
    except np.linalg.LinAlgError:
        noise_cov += np.eye(n_features) * 1e-6
        L = np.linalg.cholesky(noise_cov)

    datasets = {0: (X.copy(), y.copy())}
    for level in aug_levels:
        if level == 0:
            continue
        X_aug_list = [X.copy()]
        y_aug_list = [y.copy()]
        for _ in range(level):
            X_copy = X.copy()
            noise = np.dot(np.random.randn(n_samples, n_features), L.T)
            X_copy[numeric_cols] = X_num + noise

            nonneg_cols = ['weight', 'number', 'Plant height', 'pH', 'BD', 'OC', 'TN', 'Clay', 'sand',
                           'SL_FW', 'SS_FW', 'SR_FW', 'TL_FW', 'EL_FW', 'AL_FW']
            for col in nonneg_cols:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].clip(lower=0)
            if 'N_rate' in X_copy.columns:
                X_copy['N_rate'] = X_copy['N_rate'].round().astype(int).clip(0, 250)
            X_aug_list.append(X_copy)
            y_aug_list.append(y.copy())
        datasets[level] = (pd.concat(X_aug_list, ignore_index=True), np.concatenate(y_aug_list))
    return datasets

t0 = time.time()
aug_datasets = generate_all_augmented_datasets(X_train, y_train, numeric_cols, aug_levels, noise_factor)

# ==========================================
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1]
    },
    'CatBoost': {
        'iterations': [100, 200],
        'depth': [4, 6],
        'learning_rate': [0.03, 0.1]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'num_leaves': [15, 31],
        'learning_rate': [0.05, 0.1]
    },
    'Linear': {}
}

param_distributions = {
    'Random Forest': {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15]
    },
    'CatBoost': {
        'iterations': [100, 150, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.03, 0.07, 0.1]
    },
    'LightGBM': {
        'n_estimators': [100, 150, 200],
        'num_leaves': [15, 31, 45],
        'learning_rate': [0.05, 0.1, 0.15]
    }
}

# ==========================================
model_names = ['Linear', 'Random Forest', 'XGBoost', 'CatBoost', 'LightGBM']
best_results = {}

for name in model_names:
    print(f"\n{'=' * 40}\model: {name}\n{'=' * 40}")
    results = []
    for level in aug_levels:
        X_tr, y_tr = aug_datasets[level]
        print(f"  aug ×{level if level > 0 else 1} (sample:{len(X_tr)})")

        if name == 'Linear':
            model = LinearRegression()
            model.fit(X_tr, y_tr)
        else:

            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            base_model = None
            if name == 'Random Forest':
                base_model = RandomForestRegressor(random_state=random_state)
            elif name == 'XGBoost':
                base_model = XGBRegressor(random_state=random_state, verbosity=0,
                                          tree_method='hist', n_jobs=-1)
            elif name == 'CatBoost':
                base_model = CatBoostRegressor(random_state=random_state, verbose=0)
            elif name == 'LightGBM':
                base_model = LGBMRegressor(random_state=random_state, verbose=-1)

            if use_random_search:
                searcher = RandomizedSearchCV(base_model, param_distributions[name],
                                              n_iter=random_iter, cv=cv, scoring='neg_mean_squared_error',
                                              n_jobs=-1, random_state=random_state, verbose=0)
            else:
                searcher = GridSearchCV(base_model, param_grids[name], cv=cv,
                                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
            searcher.fit(X_tr, y_tr)
            model = searcher.best_estimator_
            print(f"      最佳参数: {searcher.best_params_}")

        pred_log = model.predict(X_test)
        y_pred = np.expm1(pred_log)
        r2 = r2_score(y_test_ori, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_ori, y_pred))
        results.append((level, r2, rmse, y_pred))
        print(f"      test R²={r2:.4f}, RMSE={rmse:.4f}")

    best = max(results, key=lambda x: (x[1], -x[2]))
    best_results[name] = best
    print(f"  best_aug: ×{best[0] if best[0] > 0 else 1}, R²={best[1]:.4f}, RMSE={best[2]:.4f}")

metrics_list = []
for name, (level, r2, rmse, _) in best_results.items():
    metrics_list.append({'model': name, 'best_aug': level if level > 0 else 1,
                         'R²': round(r2, 4), 'RMSE': round(rmse, 4)})
pd.DataFrame(metrics_list).to_excel(output_metrics, index=False)

# ==========================================
color_schemes = {
    'Linear': {'scatter': '#1f77b4', 'edge': '#0a4d8c', 'line': '#1f77b4', 'band': '#1f77b4', 'box_bg': '#d0e4ff',
               'box_edge': '#1f77b4', 'text': '#0a4d8c'},
    'Random Forest': {'scatter': '#2ca02c', 'edge': '#1b5e20', 'line': '#2ca02c', 'band': '#2ca02c',
                      'box_bg': '#d4edda', 'box_edge': '#2ca02c', 'text': '#1b5e20'},
    'XGBoost': {'scatter': '#ff7f0e', 'edge': '#b85c00', 'line': '#ff7f0e', 'band': '#ff7f0e', 'box_bg': '#ffe5cc',
                'box_edge': '#ff7f0e', 'text': '#b85c00'},
    'CatBoost': {'scatter': '#9467bd', 'edge': '#5e3c7a', 'line': '#9467bd', 'band': '#9467bd', 'box_bg': '#e8daf0',
                 'box_edge': '#9467bd', 'text': '#5e3c7a'},
    'LightGBM': {'scatter': '#17becf', 'edge': '#0e7d87', 'line': '#17becf', 'band': '#17becf', 'box_bg': '#ccf2f4',
                 'box_edge': '#17becf', 'text': '#0e7d87'}
}

fig, axes = plt.subplots(1, 5, figsize=(30, 6))
axes = axes.flatten()

titles = ['(a) Linear', '(b) Random Forest', '(c) XGBoost', '(d) CatBoost', '(e) LightGBM']

for idx, name in enumerate(model_names):
    ax = axes[idx]
    aug_level, r2, rmse, y_pred = best_results[name]
    y_true = y_test_ori
    colors = color_schemes[name]

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    line_range = [min_val - margin, max_val + margin]

    ax.plot(line_range, line_range, color=colors['line'], linestyle='--', linewidth=3, label='Perfect fit')

    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    std_err = np.sqrt(mse)
    ci = 1.96 * std_err
    x_sorted = np.sort(y_true)
    ax.fill_between(x_sorted, x_sorted - ci, x_sorted + ci,
                    color=colors['band'], alpha=0.25, label='Confidence band')

    ax.scatter(y_true, y_pred, c=colors['scatter'], s=70, alpha=0.8,
               edgecolors=colors['edge'], linewidth=2.0, zorder=5)

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(titles[idx], fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)

    textstr = f'$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}'
    props = dict(boxstyle='round,pad=0.5', facecolor=colors['box_bg'], alpha=0.9,
                 edgecolor=colors['box_edge'], linewidth=2.5)
    ax.text(0.05, 0.93, textstr, transform=ax.transAxes, fontsize=13, verticalalignment='top',
            bbox=props, color=colors['text'], weight='bold')
    ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(output_fig_pdf, dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(output_fig_png, dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
