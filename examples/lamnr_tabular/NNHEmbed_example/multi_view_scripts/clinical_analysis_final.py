import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import sys

# --- CONFIGURATION ---
DATA_DIR = Path("lamnr_repro_pack/processed/trimmed_input")
SIMLR_DATA_DIR = Path("lamnr_repro_pack/processed/latent_projections/")
RUNS_DIR = Path("runs/multiview_production")

TARGET_FILES = {
    "NNL": DATA_DIR / "aligned_targets_NNL.csv",
    "PPMI": DATA_DIR / "aligned_targets_PPMI.csv"
}

RAW_INPUTS = {
    "NNL": [SIMLR_DATA_DIR / "clean_aligned_projection_NNL_DTI.csv", 
            SIMLR_DATA_DIR / "clean_aligned_projection_NNL_rsfMRI.csv", 
            SIMLR_DATA_DIR / "clean_aligned_projection_NNL_T1.csv"],
    "PPMI": [SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_DTI.csv", 
             SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_rsfMRI.csv", 
             SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_T1.csv"]
}

# Model Definitions
BEST_LAMNR = {"NNL": "vicreg_lambda0.3", "PPMI": "vicreg_lambda2.0"}
BASELINE_LAMNR = {"NNL": "seed42_baseline", "PPMI": "seed42_baseline"}

NUISANCE_COLS = ["AGE", "PTGENDER", "PTEDUCAT"]
N_BOOTSTRAPS = 1000

def load_z(prefix):
    """Loads and averages the 3 latent views for a given run prefix."""
    zs = []
    for i in range(3):
        f = Path(f"{prefix}_whitened_view{i}.csv")
        if not f.exists(): raise FileNotFoundError(f"Missing view {i}: {f}")
        df = pd.read_csv(f).apply(pd.to_numeric, errors='coerce').dropna()
        zs.append(df.values)
    return np.mean(zs, axis=0)

def load_z_linear(file_list, n_components=31):
    """Generates the Linear PCA baseline."""
    data = [pd.read_csv(f).apply(pd.to_numeric, errors='coerce').dropna().values for f in file_list]
    X = StandardScaler().fit_transform(np.hstack(data))
    pca = PCA(n_components=min(n_components, X.shape[0]), random_state=42)
    return pca.fit_transform(X)

def get_cv_preds(X, y):
    """Generates 5-fold cross-validated predictions."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(y, dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    for train, test in kf.split(X_scaled):
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(X_scaled[train], y[train])
        preds[test] = model.predict(X_scaled[test])
    return preds

def bootstrap_delta(y_true, p_main, p_ref, n_boot=N_BOOTSTRAPS):
    """Computes Delta r, 95% CI, and p-value via bootstrapping."""
    n = len(y_true)
    r_main = pearsonr(y_true, p_main)[0]
    r_ref = pearsonr(y_true, p_ref)[0]
    base_delta = r_main - r_ref
    
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            d = pearsonr(y_true[idx], p_main[idx])[0] - pearsonr(y_true[idx], p_ref[idx])[0]
            deltas.append(d if not np.isnan(d) else 0)
        except: deltas.append(0)
    
    deltas = np.array(deltas)
    p_val = 2 * min(np.mean(deltas > 0), np.mean(deltas < 0))
    return base_delta, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5), p_val

def run():
    results = []
    for ds in ["NNL", "PPMI"]:
        print(f"Processing {ds}...")
        df_t = pd.read_csv(TARGET_FILES[ds])
        # Clean Nuisance
        for c in NUISANCE_COLS: df_t[c] = pd.to_numeric(df_t[c], errors='coerce')
        if "PTGENDER" in df_t: df_t["PTGENDER"] = df_t["PTGENDER"].astype('category').cat.codes
        
        # Load all 3 latent sets
        z_lin = load_z_linear(RAW_INPUTS[ds])
        z_lam = load_z(RUNS_DIR / ds / BEST_LAMNR[ds])
        z_none = load_z(RUNS_DIR / ds / BASELINE_LAMNR[ds])
        
        n_min = min(len(df_t), len(z_lin), len(z_lam), len(z_none))
        df_t, z_lin, z_lam, z_none = df_t.iloc[:n_min], z_lin[:n_min], z_lam[:n_min], z_none[:n_min]
        
        targets = [c for c in df_t.columns if c not in NUISANCE_COLS and "id" not in c.lower() and "unnamed" not in c.lower()]
        
        for t in targets:
            y_raw = pd.to_numeric(df_t[t], errors='coerce')
            mask = df_t[NUISANCE_COLS].notna().all(axis=1) & y_raw.notna()
            if mask.sum() < 40: continue
            
            y, X_n = y_raw[mask].values, df_t.loc[mask, NUISANCE_COLS].values
            p_lin = get_cv_preds(np.hstack([X_n, z_lin[mask]]), y)
            p_lam = get_cv_preds(np.hstack([X_n, z_lam[mask]]), y)
            p_none = get_cv_preds(np.hstack([X_n, z_none[mask]]), y)
            
            d_lin, low_lin, high_lin, p_lin_val = bootstrap_delta(y, p_lam, p_lin)
            d_none, low_none, high_none, p_none_val = bootstrap_delta(y, p_lam, p_none)
            
            results.append({
                "Dataset": ds, "Outcome": t, "N": mask.sum(),
                "Delta_Lin": d_lin, "CI_Lin": f"[{low_lin:.3f}, {high_lin:.3f}]", "p_Lin": p_lin_val,
                "Delta_None": d_none, "CI_None": f"[{low_none:.3f}, {high_none:.3f}]", "p_None": p_none_val
            })
    pd.DataFrame(results).to_csv("full_clinical_comparison.csv", index=False)
    print("Results saved to full_clinical_comparison.csv")

if __name__ == "__main__": run()