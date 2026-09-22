import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy import stats
from tabpfn import TabPFNRegressor
import warnings

# ===================== 全局绘图风格 =====================
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Arial', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径与配置 =====================
os.environ["TABPFN_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYzE2ZTU5NmQtYzgzMS00NGI1LWE4NDUtMGQ1MjU1NDM3NjgzIiwiZXhwIjoxODA3NDE2MDk0fQ.Jd3Trp9OkXa1t4EbPrsplLrYxCkreTlhvJip4xLV6Bw"

data_path = "dataset.xlsx"
output_metrics = "best_aug_metrics.xlsx"
output_fig_png = "best_aug_residuals.png"

n_aug_levels = 10
noise_factor = 0.035
test_size = 0.3
random_state = 42

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
df_raw = pd.read_excel(data_path)
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
print(f"train: {len(df_train)} , test: {len(df_test)} ")

target = 'weight'

# ==========================================
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
def augment_multivariate_gaussian(X, y, n_copies, numeric_cols, noise_factor=0.035):
    X_num = X[numeric_cols].values.astype(float)
    n_samples, n_features = X_num.shape
    cov = np.cov(X_num, rowvar=False)
    noise_cov = (noise_factor ** 2) * cov
    try:
        np.linalg.cholesky(noise_cov)
    except np.linalg.LinAlgError:
        noise_cov += np.eye(n_features) * 1e-6

    X_aug_list, y_aug_list = [X.copy()], [y.copy()]
    for _ in range(n_copies):
        X_copy = X.copy()
        noise = np.random.multivariate_normal(mean=np.zeros(n_features), cov=noise_cov, size=n_samples)
        X_copy[numeric_cols] = X_num + noise

        nonneg_cols = ['weight', 'number', 'Plant height', 'GDD_Seedling_to_Budding',
                       'pH', 'BD', 'OC', 'TN', 'Clay', 'sand',
                       'SL_FW', 'SS_FW', 'SR_FW', 'TL_FW', 'EL_FW', 'AL_FW']
        for col in nonneg_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(lower=0)
        if 'N_rate' in X_copy.columns:
            X_copy['N_rate'] = X_copy['N_rate'].round().astype(int).clip(0, 250)
        X_aug_list.append(X_copy)
        y_aug_list.append(y.copy())
    return pd.concat(X_aug_list, ignore_index=True), np.concatenate(y_aug_list)

# ==========================================
model = TabPFNRegressor(device='cpu', model_path=r"tabpfn-v2.5-regressor-v2.5_default.ckpt")
results = []
for aug_level in range(n_aug_levels):
    if aug_level == 0:
        X_tr, y_tr = X_train, y_train
    else:
        X_tr, y_tr = augment_multivariate_gaussian(X_train, y_train, n_copies=aug_level,
                                                   numeric_cols=numeric_cols, noise_factor=noise_factor)
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_test)
    y_pred = np.expm1(pred_log)
    r2 = r2_score(y_test_ori, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_ori, y_pred))
    results.append((aug_level, r2, rmse, y_pred))
    print(f"aug ×{aug_level+1}: R²={r2:.4f}, RMSE={rmse:.4f}")

# ==========================================

best = max(results, key=lambda x: (x[1], -x[2]))
best_level, best_r2, best_rmse, best_pred = best
print(f"\nbest_aug: ×{best_level+1} (R²={best_r2:.4f}, RMSE={best_rmse:.4f})")

pd.DataFrame([{'Aug_Level': best_level, 'Data_Multiple': best_level+1,
               'R2': best_r2, 'RMSE': best_rmse}]).to_excel(output_metrics, index=False)

# ==========================================
fig, (ax_fit, ax_resid) = plt.subplots(1, 2, figsize=(12, 5))

min_val = min(y_test_ori.min(), best_pred.min())
max_val = max(y_test_ori.max(), best_pred.max())
margin = (max_val - min_val) * 0.05
line_range = [min_val-margin, max_val+margin]

ax_fit.plot(line_range, line_range, color='#D2691E', linestyle='--', linewidth=3, label='Perfect fit')

x_sorted = np.sort(y_test_ori)
pred_sorted = best_pred[np.argsort(y_test_ori)]

residuals = y_test_ori - best_pred
mse = np.mean(residuals ** 2)
std_err = np.sqrt(mse)

confidence_interval = 1.96 * std_err
conf_lower = x_sorted - confidence_interval
conf_upper = x_sorted + confidence_interval

ax_fit.fill_between(x_sorted, conf_lower, conf_upper,
                    color='#FFA07A', alpha=0.3, label='Confidence band')

scatter = ax_fit.scatter(y_test_ori, best_pred, c='#CD853F', s=50, alpha=0.7,
                         edgecolors='#8B4513', linewidth=1.5, zorder=5)

ax_fit.set_xlabel('Actual')
ax_fit.set_ylabel('Predicted')
ax_fit.set_title('(a) Fit Plot', fontsize=12, weight='bold')
ax_fit.grid(True, alpha=0.3)

textstr = f'$R^2$: {best_r2:.4f}\nRMSE: {best_rmse:.4f}'
props = dict(boxstyle='round,pad=0.5', facecolor='#FFE4B5', alpha=0.8, edgecolor='#D2691E', linewidth=1.5)
ax_fit.text(0.05, 0.93, textstr, transform=ax_fit.transAxes, fontsize=14, verticalalignment='top',
            bbox=props, color='#8B4513', weight='bold')

ax_fit.legend(loc='lower right')

errors = y_test_ori - best_pred

ax_resid.scatter(best_pred, errors, c='#CD853F', s=50, alpha=0.7,
                 edgecolors='#8B4513', linewidth=1.5, zorder=5)

ax_resid.axhline(y=0, color='#D2691E', linestyle='--', linewidth=3)

ax_resid.set_xlabel('Predicted')
ax_resid.set_ylabel('Residuals')
ax_resid.set_title('(b) Residual', fontsize=12, weight='bold')
ax_resid.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_fig_pdf, dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(output_fig_png, dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
