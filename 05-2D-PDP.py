import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from tabpfn import TabPFNRegressor
import warnings

warnings.filterwarnings('ignore')

# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

model_path = r"tabpfn-v2.5-regressor-v2.5_default.ckpt"
data_path = r"dataset.xlsx"
save_dir = r"pdp_analysis_results"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "2D_Interactions_Weather.png")

random_state = 42
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

WEATHER_FEATURES = [
    'S_prec', 'T_prec', 'E_prec', 'A_prec', 'M_prec',
    'S_srad', 'T_srad', 'E_srad', 'A_srad', 'M_srad',
    'S_GDD', 'S-T_GDD', 'S-E_GDD', 'S-A_GDD', 'S-M_GDD'
]

target = 'weight'

# ==========================================
df_raw = pd.read_excel(data_path)
df_raw['env_key'] = df_raw['year'].astype(str) + '_' + df_raw['region'].astype(str)

train_dfs = []
for key, group in df_raw.groupby('env_key'):
    if len(group) < 2:
        train_dfs.append(group)
    else:
        tr, _ = train_test_split(group, test_size=0.3, random_state=random_state)
        train_dfs.append(tr)
df_train = pd.concat(train_dfs, ignore_index=True)

feature_cols_used = [col for col in FEATURE_COLS if col != 'number']
X_train = df_train[feature_cols_used].copy()
y_train_ori = df_train[target].values

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))
X_train = X_train.fillna(X_train.mean(numeric_only=True))
y_train_log = np.log1p(y_train_ori)

# ==========================================
noise_factor = 0.035
n_copies = 1

def augment_multivariate_gaussian(X, n_copies, noise_factor):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].values.astype(float)
    n_samples, n_features = X_num.shape
    cov = np.cov(X_num, rowvar=False)
    noise_cov = (noise_factor ** 2) * cov
    try:
        np.linalg.cholesky(noise_cov)
    except np.linalg.LinAlgError:
        noise_cov += np.eye(n_features) * 1e-6

    X_aug_list = [X.copy()]
    for _ in range(n_copies):
        X_copy = X.copy()
        noise = np.random.multivariate_normal(mean=np.zeros(n_features), cov=noise_cov, size=n_samples)
        X_copy[numeric_cols] = X_num + noise
        nonneg_cols = [
            'Plant height', 'Number of main stems',
            'SL_FW', 'SS_FW', 'SR_FW', 'SL_DW', 'SS_DW', 'SR_DW',
            'TL_FW', 'TS_FW', 'TR_FW', 'TL_DW', 'TS_DW', 'TR_DW',
            'EL_FW', 'ES_FW', 'ER_FW', 'ES_DW', 'EL_DW', 'ER_DW',
            'AL_FW', 'AS_FW', 'AR_FW', 'AS_DW', 'AL_DW', 'AR_DW',
            'AN', 'BD', 'OC', 'TN', 'Clay', 'pH', 'sand',
            'S_prec', 'T_prec', 'E_prec', 'A_prec', 'M_prec',
            'S_srad', 'T_srad', 'E_srad', 'A_srad', 'M_srad',
            'S_GDD', 'S-T_GDD', 'S-E_GDD', 'S-A_GDD', 'S-M_GDD'
        ]
        for col in nonneg_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(lower=0)
        if 'N_rate' in X_copy.columns:
            X_copy['N_rate'] = X_copy['N_rate'].round().astype(int).clip(0, 250)
        X_aug_list.append(X_copy)
    return pd.concat(X_aug_list, ignore_index=True)

X_aug = augment_multivariate_gaussian(X_train, n_copies=n_copies, noise_factor=noise_factor)
y_aug = np.concatenate([y_train_log, y_train_log])

# ==========================================
model = TabPFNRegressor(device='cuda', model_path=model_path)
model.fit(X_aug, y_aug)

# ==========================================
valid_weather = [f for f in WEATHER_FEATURES if f in X_aug.columns]
pdp_scores = {}
for feat in valid_weather:
    try:
        pdp_display = PartialDependenceDisplay.from_estimator(
            model, X_aug, [feat],
            grid_resolution=20,
            subsample=100,
            kind='average',
            n_jobs=1
        )
        y_vals = pdp_display.axes_[0, 0].lines[0].get_ydata()
        score = np.std(y_vals)
        pdp_scores[feat] = score
        plt.close()
    except Exception as e:
        print(f"feature {feat} failure: {e}")

sorted_scores = sorted(pdp_scores.items(), key=lambda x: x[1], reverse=True)
top6_weather = [item[0] for item in sorted_scores[:6]]

# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for idx, feat in enumerate(top6_weather):
    ax = axes[idx]
    feat1 = 'N_rate'
    feat2 = feat

    title = f"{sub_labels[idx]} Interaction: {feat1} vs {feat2}"

    grid_size = 20
    x1 = X_aug[feat1].values
    x2 = X_aug[feat2].values
    x1_grid = np.linspace(x1.min(), x1.max(), grid_size)
    x2_grid = np.linspace(x2.min(), x2.max(), grid_size)
    X1_inv, X2_inv = np.meshgrid(x1_grid, x2_grid)

    X_grid = pd.DataFrame(np.tile(X_aug.mean().values, (len(X1_inv.ravel()), 1)), columns=X_aug.columns)
    X_grid[feat1] = X1_inv.ravel()
    X_grid[feat2] = X2_inv.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(X_grid)

    if isinstance(preds, tuple):
        predictions = preds[0].cpu().numpy()
    elif hasattr(preds, 'cpu'):
        predictions = preds.cpu().numpy()
    else:
        predictions = preds
    predictions = predictions.reshape(X1_inv.shape)

    levels_cf = np.linspace(predictions.min(), predictions.max(), 100)
    cf = ax.contourf(X1_inv, X2_inv, predictions, levels=levels_cf, cmap='RdYlBu_r', extend='both')

    q1 = np.percentile(predictions, 25)
    median = np.percentile(predictions, 50)
    q3 = np.percentile(predictions, 75)
    ax.contour(X1_inv, X2_inv, predictions, levels=[q1], colors='#1f77b4', linestyles='dashed', linewidths=2.5)
    ax.contour(X1_inv, X2_inv, predictions, levels=[median], colors='#2ca02c', linestyles='solid', linewidths=2.5)
    ax.contour(X1_inv, X2_inv, predictions, levels=[q3], colors='#d62728', linestyles='dashed', linewidths=2.5)

    min_idx = np.unravel_index(np.argmin(predictions), predictions.shape)
    max_idx = np.unravel_index(np.argmax(predictions), predictions.shape)
    ax.scatter(X1_inv[max_idx], X2_inv[max_idx], marker='*', s=250, facecolor='gold', edgecolor='black', linewidth=2.5, zorder=10)
    ax.scatter(X1_inv[min_idx], X2_inv[min_idx], marker='o', s=120, facecolor='navy', edgecolor='white', linewidth=2.5, zorder=10)

    cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.02)
    cb_ticks = np.linspace(predictions.min(), predictions.max(), 5)
    cb.set_ticks(cb_ticks)
    cb.ax.set_yticklabels([f"{val:.2f}" for val in cb_ticks], fontsize=9, fontweight='bold')
    cb.outline.set_linewidth(2.0)

    ax.set_xlabel(feat1, fontsize=12, fontweight='bold')
    ax.set_ylabel(feat2, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(labelsize=10, width=2.5)
    ax.set_xlim(X1_inv.min() - 0.05*(X1_inv.max()-X1_inv.min()), X1_inv.max() + 0.05*(X1_inv.max()-X1_inv.min()))
    ax.set_ylim(X2_inv.min() - 0.05*(X2_inv.max()-X2_inv.min()), X2_inv.max() + 0.05*(X2_inv.max()-X2_inv.min()))
    ax.set_aspect('auto')

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
