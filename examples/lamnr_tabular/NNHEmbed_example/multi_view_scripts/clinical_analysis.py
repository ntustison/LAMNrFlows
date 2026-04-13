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
# Vos chemins personnalisés pour SiMLR (conservés)
SIMLR_DATA_DIR = Path("lamnr_repro_pack/processed/latent_projections/")
RUNS_DIR = Path("runs/multiview_production")

# Fichiers Cibles
TARGET_FILES = {
    "NNL": DATA_DIR / "aligned_targets_NNL.csv",
    "PPMI": DATA_DIR / "aligned_targets_PPMI.csv"
}

# Inputs pour la Baseline Linéaire
# (J'utilise vos chemins modifiés pointant vers SIMLR_DATA_DIR)
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

# Modèles LAMNr Gagnants (Préfixes de fichiers)
LAMNR_MODELS = {
    "NNL": RUNS_DIR / "NNL" / "vicreg_lambda0.3", 
    "PPMI": RUNS_DIR / "PPMI" / "vicreg_lambda2.0" 
}

NUISANCE_COLS = ["AGE", "PTGENDER", "PTEDUCAT"]

# --- CORRECTION ICI ---
def load_z_lamnr(model_prefix):
    """Charge et moyenne les 3 vues latentes en utilisant le préfixe de fichier."""
    zs = []
    prefix_str = str(model_prefix) # Ex: runs/.../vicreg_lambda0.3
    
    print(f"   Chargement depuis le préfixe : {prefix_str}")
    
    for i in range(3):
        # Construction directe du chemin de fichier
        # On ajoute "_whitened_viewX.csv" au préfixe
        file_path = Path(f"{prefix_str}_whitened_view{i}.csv")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Vue {i} introuvable : {file_path}")
            
        # Lecture (sans header car c'est une matrice brute)
        df = pd.read_csv(file_path, header=None)
        zs.append(df.values)
    
    # Moyenne des vues (Consensus Latent)
    return np.mean(zs, axis=0)
# ----------------------

def load_z_linear(file_list, n_components=31):
    """Baseline Linéaire (PCA sur concaténation ou chargement direct SiMLR)."""
    data_list = []
    for f in file_list:
        if not f.exists():
             # Fallback: si le fichier SiMLR n'existe pas, on cherche dans DATA_DIR standard
             fallback = DATA_DIR / f.name
             if fallback.exists():
                 print(f"   [INFO] Fichier SiMLR non trouvé, repli sur : {fallback}")
                 f = fallback
             else:
                 raise FileNotFoundError(f"Input manquant : {f}")
        
        df = pd.read_csv(f)
        # Si le fichier contient des headers (strings), on tente de convertir
        try:
            vals = df.values.astype(float)
        except ValueError:
             # Probablement un header présent, on relit sans header ou on skip la 1ère ligne ?
             # Supposons que c'est propre (comme généré par les scripts précédents)
             # Si échec, c'est peut-être qu'il y a des colonnes non-numériques
             vals = df.select_dtypes(include=[np.number]).values
             
        data_list.append(vals)
    
    # Concaténation
    X_concat = np.hstack(data_list)
    
    # PCA (SiMLR proxy)
    # Si les données sont déjà réduites (31 dims), la PCA va juste les tourner/nettoyer
    scaler = StandardScaler()
    X_concat = scaler.fit_transform(X_concat)
    
    pca = PCA(n_components=n_components, random_state=42)
    # Si n_samples < n_components, PCA râle. On limite n_components.
    n_samples = X_concat.shape[0]
    if n_samples < n_components:
        pca = PCA(n_components=n_samples, random_state=42)
        
    return pca.fit_transform(X_concat)

def evaluate_prediction(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(y)
    X = StandardScaler().fit_transform(X)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        model.fit(X_train, y_train)
        preds[test_idx] = model.predict(X_test)
        
    try:
        corr, _ = pearsonr(y, preds)
        return corr
    except:
        return 0.0

def run_analysis():
    print("--- Analyse Clinique : LAMNr vs Linear (SiMLR) ---")
    
    results = []

    for dataset in ["NNL", "PPMI"]:
        print(f"\nTraitement : {dataset}")
        
        # 1. Chargement Cibles
        if not TARGET_FILES[dataset].exists():
            print(f"[SKIP] Cibles manquantes : {TARGET_FILES[dataset]}")
            continue
            
        df_target = pd.read_csv(TARGET_FILES[dataset])
        
        # Encodage
        if df_target["PTGENDER"].dtype == 'object':
            df_target["PTGENDER"] = df_target["PTGENDER"].astype('category').cat.codes
        if df_target["PTEDUCAT"].dtype == 'object':
             df_target["PTEDUCAT"] = pd.to_numeric(df_target["PTEDUCAT"], errors='coerce')
        if df_target["AGE"].dtype == 'object':
             df_target["AGE"] = pd.to_numeric(df_target["AGE"], errors='coerce')

        # 2. Chargement Images
        try:
            print("   Chargement Baseline Linéaire...")
            Z_lin = load_z_linear(RAW_INPUTS[dataset])
            
            print(f"   Chargement LAMNr...")
            Z_lam = load_z_lamnr(LAMNR_MODELS[dataset])
        except Exception as e:
            print(f"   [ERREUR CHARGEMENT] {e}")
            continue
        
        # Alignement Dimensions
        n_min = min(len(df_target), len(Z_lin), len(Z_lam))
        if len(df_target) != n_min:
            print(f"   [INFO] Troncature commune à {n_min} sujets.")
            df_target = df_target.iloc[:n_min]
            Z_lin = Z_lin[:n_min]
            Z_lam = Z_lam[:n_min]

        # 3. Boucle Cibles
        potential_targets = [c for c in df_target.columns 
                             if c not in NUISANCE_COLS 
                             and "id" not in c.lower() 
                             and "index" not in c.lower()
                             and "unnamed" not in c.lower()]
        
        print(f"   Analyse de {len(potential_targets)} variables cibles...")
        
        for target_name in potential_targets:
            y_raw = pd.to_numeric(df_target[target_name], errors='coerce')
            
            # DataFrame temp pour dropna conjoint
            temp_df = df_target[NUISANCE_COLS].copy()
            temp_df["TARGET"] = y_raw
            
            valid_mask = temp_df.notna().all(axis=1)
            n_valid = valid_mask.sum()
            
            if n_valid < 40:
                continue
                
            y = temp_df.loc[valid_mask, "TARGET"].values
            X_nuis = temp_df.loc[valid_mask, NUISANCE_COLS].values
            Z_lin_sub = Z_lin[valid_mask]
            Z_lam_sub = Z_lam[valid_mask]
            
            # Scores
            r_nuis = evaluate_prediction(X_nuis, y)
            r_lin = evaluate_prediction(np.hstack([X_nuis, Z_lin_sub]), y)
            r_lam = evaluate_prediction(np.hstack([X_nuis, Z_lam_sub]), y)
            
            uplift = r_lam - r_lin
            
            results.append({
                "Dataset": dataset,
                "Target": target_name,
                "N": n_valid,
                "R_Nuisance": r_nuis,
                "R_Linear": r_lin,
                "R_LAMNr": r_lam,
                "Uplift": uplift
            })
            
            print(f"   -> {target_name:<20} | N={n_valid:<3} | Lin:{r_lin:.3f} | LAMNr:{r_lam:.3f} | Up:{uplift:+.3f}")

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv("clinical_results_final.csv", index=False)
        print("\n=== TOP UPLIFTS (LAMNr - Linear) ===")
        print(df_res.sort_values("Uplift", ascending=False).head(10)[["Dataset", "Target", "Uplift", "R_LAMNr"]].to_string(index=False))
    else:
        print("\nAucun résultat.")

if __name__ == "__main__":
    run_analysis()