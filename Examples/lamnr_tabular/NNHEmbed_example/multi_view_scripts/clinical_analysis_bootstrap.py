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

# Files
TARGET_FILES = {
    "NNL": DATA_DIR / "aligned_targets_NNL.csv",
    "PPMI": DATA_DIR / "aligned_targets_PPMI.csv"
}

RAW_INPUTS = {
    "NNL": [
        SIMLR_DATA_DIR / "clean_aligned_projection_NNL_DTI.csv", 
        SIMLR_DATA_DIR / "clean_aligned_projection_NNL_rsfMRI.csv", 
        SIMLR_DATA_DIR / "clean_aligned_projection_NNL_T1.csv"
    ],
    "PPMI": [
        SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_DTI.csv", 
        SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_rsfMRI.csv", 
        SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_T1.csv"
    ]
}

# Best Models
LAMNR_MODELS = {
    "NNL": RUNS_DIR / "NNL" / "vicreg_lambda0.3", 
    "PPMI": RUNS_DIR / "PPMI" / "vicreg_lambda2.0" 
}

NUISANCE_COLS = ["AGE", "PTGENDER", "PTEDUCAT"]
N_BOOTSTRAPS = 1000  # Number of resamples

# --- HELPERS ---

def load_z_lamnr(model_prefix):
    zs = []
    prefix_str = str(model_prefix)
    for i in range(3):
        f = Path(f"{prefix_str}_whitened_view{i}.csv")
        if not f.exists(): raise FileNotFoundError(f"Missing: {f}")
        df = pd.read_csv(f)
        # Robust loading
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        zs.append(df.values)
    return np.mean(zs, axis=0)

def load_z_linear(file_list, n_components=31):
    data_list = []
    for f in file_list:
        if not f.exists():
             fallback = DATA_DIR / f.name
             if fallback.exists(): f = fallback
             else: raise FileNotFoundError(f"Missing: {f}")
        df = pd.read_csv(f)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        data_list.append(df.values)
    X_concat = np.hstack(data_list)
    scaler = StandardScaler()
    X_concat = scaler.fit_transform(X_concat)
    pca = PCA(n_components=n_components, random_state=42)
    if X_concat.shape[0] < n_components:
        pca = PCA(n_components=X_concat.shape[0], random_state=42)
    return pca.fit_transform(X_concat)

def get_predictions(X, y):
    """
    Returns the vector of cross-validated predictions (y_pred) for the whole dataset.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(y, dtype=float)
    X = StandardScaler().fit_transform(X)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        model.fit(X_train, y_train)
        preds[test_idx] = model.predict(X_test)
        
    return preds

def bootstrap_stats(y_true, y_pred_lin, y_pred_lam, n_boot=N_BOOTSTRAPS):
    """
    Bootstraps the difference in correlation (r_lam - r_lin).
    Returns: delta, ci_lower, ci_upper, p_value
    """
    n_samples = len(y_true)
    deltas = []
    
    # Calculate base correlations
    try:
        base_r_lin, _ = pearsonr(y_true, y_pred_lin)
        base_r_lam, _ = pearsonr(y_true, y_pred_lam)
    except:
        return 0, 0, 0, 1.0
        
    base_delta = base_r_lam - base_r_lin
    
    rng = np.random.RandomState(42)
    
    for _ in range(n_boot):
        # Resample indices with replacement
        indices = rng.randint(0, n_samples, n_samples)
        
        y_b = y_true[indices]
        p_lin_b = y_pred_lin[indices]
        p_lam_b = y_pred_lam[indices]
        
        # Calculate r on bootstrap sample (handle constant input case)
        try:
            r_lin_b, _ = pearsonr(y_b, p_lin_b)
            r_lam_b, _ = pearsonr(y_b, p_lam_b)
            
            # Check for NaN (if resampling creates constant vector)
            if np.isnan(r_lin_b): r_lin_b = 0
            if np.isnan(r_lam_b): r_lam_b = 0
                
            deltas.append(r_lam_b - r_lin_b)
        except:
            deltas.append(0)
            
    deltas = np.array(deltas)
    
    # Confidence Interval (2.5% - 97.5%)
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    
    # P-value (Two-sided)
    # Fraction of bootstraps that cross 0 or are opposite sign to base_delta
    # A simple approximation for p-value: 2 * min(P(d>0), P(d<0))
    # Correcting for discrete nature
    p_gt_0 = np.mean(deltas > 0)
    p_lt_0 = np.mean(deltas < 0)
    p_val = 2 * min(p_gt_0, p_lt_0)
    
    return base_delta, ci_lower, ci_upper, p_val

def run_analysis():
    print("--- Clinical Analysis with Bootstrapping (95% CI & p-values) ---")
    
    results = []

    for dataset in ["NNL", "PPMI"]:
        print(f"\nProcessing: {dataset}")
        
        if not TARGET_FILES[dataset].exists():
            continue
            
        df_target = pd.read_csv(TARGET_FILES[dataset])
        
        # Encoding
        if df_target["PTGENDER"].dtype == 'object':
            df_target["PTGENDER"] = df_target["PTGENDER"].astype('category').cat.codes
        df_target["PTEDUCAT"] = pd.to_numeric(df_target["PTEDUCAT"], errors='coerce')
        df_target["AGE"] = pd.to_numeric(df_target["AGE"], errors='coerce')

        # Load Latents
        try:
            print("   Loading Linear Baseline...")
            Z_lin = load_z_linear(RAW_INPUTS[dataset])
            print("   Loading LAMNr...")
            Z_lam = load_z_lamnr(LAMNR_MODELS[dataset])
        except Exception as e:
            print(f"   [ERROR] {e}")
            continue
        
        # Align
        n_min = min(len(df_target), len(Z_lin), len(Z_lam))
        df_target = df_target.iloc[:n_min]
        Z_lin = Z_lin[:n_min]
        Z_lam = Z_lam[:n_min]

        # Targets
        potential_targets = [c for c in df_target.columns 
                             if c not in NUISANCE_COLS 
                             and "id" not in c.lower() 
                             and "index" not in c.lower()
                             and "unnamed" not in c.lower()]
        
        for target_name in potential_targets:
            y_raw = pd.to_numeric(df_target[target_name], errors='coerce')
            temp_df = df_target[NUISANCE_COLS].copy()
            temp_df["TARGET"] = y_raw
            
            valid_mask = temp_df.notna().all(axis=1)
            n_valid = valid_mask.sum()
            
            if n_valid < 40: continue
                
            y = temp_df.loc[valid_mask, "TARGET"].values
            X_nuis = temp_df.loc[valid_mask, NUISANCE_COLS].values
            Z_lin_sub = Z_lin[valid_mask]
            Z_lam_sub = Z_lam[valid_mask]
            
            # 1. Get Predictions (CV)
            preds_lin = get_predictions(np.hstack([X_nuis, Z_lin_sub]), y)
            preds_lam = get_predictions(np.hstack([X_nuis, Z_lam_sub]), y)
            
            # 2. Bootstrap Stats
            delta, ci_low, ci_high, p_val = bootstrap_stats(y, preds_lin, preds_lam)
            
            # Calculate base R (just for display)
            r_lin, _ = pearsonr(y, preds_lin)
            r_lam, _ = pearsonr(y, preds_lam)
            
            results.append({
                "Dataset": dataset,
                "Outcome": target_name,
                "N": n_valid,
                "R_Linear": r_lin,
                "R_LAMNr": r_lam,
                "Delta": delta,
                "CI_Lower": ci_low,
                "CI_Upper": ci_high,
                "p_value": p_val
            })
            
            print(f"   -> {target_name:<20} | Delta={delta:+.3f} | 95% CI=[{ci_low:.3f}, {ci_high:.3f}] | p={p_val:.4f}")

    # Export
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv("clinical_bootstrap_results.csv", index=False)
        print("\n--- Finished. Saved to 'clinical_bootstrap_results.csv' ---")

if __name__ == "__main__":
    run_analysis()