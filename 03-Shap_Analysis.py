import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import shap
import warnings

warnings.filterwarnings('ignore')

# ==========================================
file_path = r"dataset.xlsx"
save_dir = r"shap_analysis_weight"
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

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

# ==========================================
df_raw = pd.read_excel(file_path)
target = 'weight'

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
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

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
model_path = r"/root/tabpfn-v2.5-regressor-v2.5_default.ckpt"
model = TabPFNRegressor(device='cuda', model_path=model_path)
model.fit(X_aug, y_aug)

# ==========================================
def create_shap_analysis(model, X_full, y_target, feature_names):
    def predict_fn(x):
        if isinstance(x, list):
            x = np.array(x)
        return model.predict(x).flatten()

    n_background = min(50, len(X_full))
    X_background_sampled = shap.sample(X_full, n_background, random_state=42)
    explainer = shap.KernelExplainer(predict_fn, X_background_sampled)
    shap_values = explainer.shap_values(X_full, nsamples=100)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': mean_abs_shap})
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

    feature_show_max = 12
    top_feat_imp_df = feat_imp_df.iloc[:feature_show_max].copy()

    if isinstance(X_full, pd.DataFrame):
        X_plot = X_full.values.copy()
    else:
        X_plot = X_full.copy()

    y_target_flat = y_target.flatten() if hasattr(y_target, 'flatten') else np.array(y_target)

    cmap_name = "RdYlBu_r"
    color_map = plt.get_cmap(cmap_name)
    target_vmin = np.percentile(y_target_flat, 5)
    target_vmax = np.percentile(y_target_flat, 95)

    fig = plt.figure(figsize=(16, 10))
    grid = gridspec.GridSpec(3, 4, figure=fig, wspace=0.55, hspace=0.40)

    ax_summary = fig.add_subplot(grid[:, :2])
    y_positions = range(len(top_feat_imp_df) - 1, -1, -1)
    ax_summary.set_yticks(y_positions)
    ax_summary.set_yticklabels(top_feat_imp_df["feature"], fontsize=10, weight='bold')
    ax_summary.set_xlabel("SHAP Value", fontsize=12, weight='bold')
    ax_summary.axvline(x=0, color="black", linewidth=1.5, alpha=0.7)
    ax_summary.grid(axis='x', linestyle='--', alpha=0.6, linewidth=1.2)

    ax_bar = ax_summary.twiny()
    ax_bar.barh(y_positions, top_feat_imp_df["importance"], color='lightgray', alpha=0.5, height=0.7)
    vals = top_feat_imp_df["importance"].values
    total_val = vals.sum()
    percent_vals = vals / total_val * 100
    x_anchor = vals.max() * 1.05
    for y_i, pct in zip(y_positions, percent_vals):
        ax_bar.text(x_anchor * 0.01, y_i, f"{pct:.1f}%", ha="left", va="center", fontsize=12, color="black", weight='bold')
    ax_bar.set_xlim(0, x_anchor * 1.15)
    ax_bar.set_xlabel("Mean Absolute SHAP Value", fontsize=12, weight='bold')
    ax_bar.tick_params(axis='x', labelsize=10)
    ax_bar.grid(False)

    for idx, feat in enumerate(top_feat_imp_df["feature"]):
        if feat not in feature_names:
            continue
        orig_id = feature_names.index(feat)
        shap_col_vals = shap_values[:, orig_id]
        feat_raw_vals = X_plot[:, orig_id]
        y_pos = len(top_feat_imp_df) - 1 - idx
        jitter = np.random.normal(0, 0.08, size=len(shap_col_vals))
        fmin, fmax = feat_raw_vals.min(), feat_raw_vals.max()
        if fmax > fmin:
            norm = plt.Normalize(vmin=fmin, vmax=fmax)
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        colors = color_map(norm(feat_raw_vals))
        ax_summary.scatter(shap_col_vals, y_pos + jitter, c=colors, s=30, alpha=0.9)  # 移除 edgecolor

    cax_legend = fig.add_axes([0.15, 0.02, 0.3, 0.02])
    norm_legend = plt.Normalize(vmin=0, vmax=1)
    sm_legend = plt.cm.ScalarMappable(norm=norm_legend, cmap=color_map)
    sm_legend.set_array([])
    cbar_legend = fig.colorbar(sm_legend, cax=cax_legend, orientation='horizontal')
    cbar_legend.set_label('Feature Value (Low → High)', fontsize=10, weight='bold')
    cbar_legend.ax.tick_params(labelsize=8)

    ax_summary.text(-0.12, 1.05, '(a)', transform=ax_summary.transAxes, fontsize=16, weight='bold', va='center')

    dep_axes = [fig.add_subplot(grid[i, j + 2]) for i in range(3) for j in range(2)]
    top6_features = top_feat_imp_df["feature"].head(6).tolist()
    sub_labels = ['(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

    for idx, feat in enumerate(top6_features):
        if idx >= len(dep_axes):
            break
        ax_dep = dep_axes[idx]
        if feat not in feature_names:
            continue
        f_id = feature_names.index(feat)
        x_val = X_plot[:, f_id]
        y_val = shap_values[:, f_id]

        scat = ax_dep.scatter(x_val, y_val, c=y_target_flat, cmap=color_map,
                              vmin=target_vmin, vmax=target_vmax,
                              s=30, alpha=0.9)

        p = ax_dep.get_position()
        cax_dep = fig.add_axes([p.x1 + 0.002, p.y0 + (p.height * (1 - 1.0)) / 2, 0.005, p.height * 1.0])
        cb = fig.colorbar(scat, cax=cax_dep)
        cb.outline.set_visible(False)
        cb.ax.tick_params(axis='y', length=1, labelsize=8)
        cb.ax.set_ylabel('')

        ax_dep.set_xlabel(feat, fontsize=10, weight='bold')
        ax_dep.set_ylabel("SHAP Value", fontsize=10, weight='bold')
        ax_dep.axhline(y=0, color="gray", linestyle="--", linewidth=1.5)
        ax_dep.tick_params(axis='both', labelsize=8)
        ax_dep.text(-0.18, 1.05, sub_labels[idx], transform=ax_dep.transAxes, fontsize=16, weight='bold', va='center')

    plt.suptitle("SHAP Analysis", fontsize=16, fontweight='bold')
    save_path = os.path.join(save_dir, 'SHAP_Report.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return shap_values

# ==========================================

shap_values = create_shap_analysis(
    model=model,
    X_full=X_aug,
    y_target=y_aug,
    feature_names=X_aug.columns.tolist()
)