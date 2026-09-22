import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from tabpfn import TabPFNRegressor
import shap
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings

warnings.filterwarnings('ignore')

# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

# ==========================================
model_path = r"tabpfn-v2.5-regressor-v2.5_default.ckpt"
save_dir = "final_figures"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "PDP_SHAP_weight.png")

# ==========================================
data_path = r"dataset.xlsx"
target = 'weight'
plot_feat = "N_rate"
random_state = 42
np.random.seed(random_state)

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
explainer = shap.KernelExplainer(model.predict, shap.sample(X_aug, 50, random_state=42))
shap_vals = explainer.shap_values(X_aug, nsamples=100)
feat_names = X_aug.columns.tolist()
feat_idx = feat_names.index(plot_feat)

# ==========================================
fig = plt.figure(figsize=(14, 6.0))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

ax1 = fig.add_subplot(gs[0])
PartialDependenceDisplay.from_estimator(
    model, X_train, [plot_feat],
    kind='both', subsample=100, grid_resolution=80, ax=ax1,
    line_kw={'color': 'gray', 'alpha': 0.3, 'linewidth': 0.8},
    pd_line_kw={'color': '#943135', 'linewidth': 3.5}
)

ymin, ymax = ax1.get_ylim()
y_range = ymax - ymin
ax1.set_ylim(ymin - 0.2 * y_range, ymax + 0.2 * y_range)

ymin_new, ymax_new = ax1.get_ylim()
rug_vals = np.random.choice(X_train[plot_feat].values, min(200, len(X_train)), replace=False)
rug_y = ymin_new - 0.05 * (ymax_new - ymin_new)
ax1.vlines(rug_vals, rug_y, rug_y + 0.02 * (ymax_new - ymin_new), color='black', alpha=0.5, linewidth=1)

ax1.set_xlabel(f'{plot_feat} (kg/ha)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Predicted Value (log scale)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Response Curve: N_rate', fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.4)
for spine in ax1.spines.values():
    spine.set_linewidth(2.5)

leg = [
    Line2D([0], [0], color='#943135', lw=3.5, label='Partial Dependence'),
    Line2D([0], [0], color='gray', lw=0.8, alpha=0.5, label='Individual Responses')
]
ax1.legend(handles=leg, loc='upper right', frameon=True, edgecolor='black', fontsize=10)

ax2 = fig.add_subplot(gs[1])
x_vals = X_aug[plot_feat].values
sp = shap_vals[:, feat_idx]
sc2 = ax2.scatter(x_vals, sp, c=y_aug, cmap='RdYlBu_r', s=35, alpha=0.7, edgecolor='white', linewidth=0.5)

low = lowess(sp, x_vals, frac=0.3, it=3)
ax2.plot(low[:, 0], low[:, 1], color='#2C3E50', linewidth=2.5)

cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.outline.set_linewidth(2)
cbar2.set_label('Observed Value (log scale)', fontsize=11, rotation=270, labelpad=16)

ax2.set_xlabel(f'{plot_feat} (kg/ha)', fontsize=13, fontweight='bold')
ax2.set_ylabel('SHAP Value', fontsize=12, fontweight='bold')
ax2.set_title('(b) Dependence Plot: N_rate', fontsize=16, fontweight='bold', pad=15)
ax2.grid(True, linestyle='--', alpha=0.4)
for spine in ax2.spines.values():
    spine.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
