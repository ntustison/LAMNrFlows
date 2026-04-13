import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

# --- CONFIGURATION ---
DATA_DIR = Path("lamnr_repro_pack/processed/trimmed_input")
SIMLR_DATA_DIR = Path("lamnr_repro_pack/processed/latent_projections/")
RUNS_DIR = Path("runs/multiview_production")

# Cibles (PPMI seulement car NNL n'a pas de catégories claires)
TARGET_FILE = DATA_DIR / "aligned_targets_PPMI.csv"

# Inputs (Linear Baseline)
RAW_INPUTS = [
    SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_DTI.csv", 
    SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_rsfMRI.csv", 
    SIMLR_DATA_DIR / "clean_aligned_projection_PPMI_T1.csv"
]

# Modèle LAMNr Gagnant
LAMNR_MODEL = RUNS_DIR / "PPMI" / "vicreg_lambda2.0"

# Nuisance
NUISANCE_COLS = ["AGE", "PTGENDER", "PTEDUCAT"]

# --- CHARGEMENT ROBUSTE (Même que précédemment) ---
def load_z_lamnr(model_prefix):
    zs = []
    prefix_str = str(model_prefix)
    print(f"   Chargement LAMNr : {prefix_str}")
    for i in range(3):
        f = Path(f"{prefix_str}_whitened_view{i}.csv")
        if not f.exists(): raise FileNotFoundError(f"Manquant : {f}")
        
        # Gestion Header/String
        df = pd.read_csv(f)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        zs.append(df.values)
    return np.mean(zs, axis=0)

def load_z_linear(file_list, n_components=31):
    data_list = []
    for f in file_list:
        if not f.exists():
             fallback = DATA_DIR / f.name
             if fallback.exists(): f = fallback
             else: raise FileNotFoundError(f"Input manquant : {f}")
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

def evaluate_classification(X, y):
    """
    Classification Logistique avec Stratified 5-Fold CV.
    Retourne (Accuracy, AUC Macro).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    auc_scores = []
    
    # Scaling indispensable pour LogisticRegression
    X = StandardScaler().fit_transform(X)
    
    # Détection multiclasse
    n_classes = len(np.unique(y))
    multi_class = 'ovr' # One-vs-Rest est robuste
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # LogisticRegressionCV trouve le meilleur C (régularisation)
        clf = LogisticRegressionCV(cv=3, multi_class=multi_class, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        
        acc_scores.append(accuracy_score(y_test, y_pred))
        
        try:
            if n_classes == 2:
                # Binaire : on prend la proba de la classe 1
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                # Multi : Macro-average
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            auc_scores.append(auc)
        except ValueError:
            pass 
            
    return np.mean(acc_scores), np.mean(auc_scores)

def run_analysis():
    print("--- Classification : PPMI Diagnosis (joinedDX) ---")
    
    if not TARGET_FILE.exists():
        print(f"[ERREUR] Fichier cible introuvable : {TARGET_FILE}")
        return

    df_target = pd.read_csv(TARGET_FILE)
    
    # Nettoyage Nuisance
    if df_target["PTGENDER"].dtype == 'object':
        df_target["PTGENDER"] = df_target["PTGENDER"].astype('category').cat.codes
    df_target["PTEDUCAT"] = pd.to_numeric(df_target["PTEDUCAT"], errors='coerce')
    df_target["AGE"] = pd.to_numeric(df_target["AGE"], errors='coerce')

    # Chargement Latents
    try:
        print("   Chargement Baseline (SiMLR)...")
        Z_lin = load_z_linear(RAW_INPUTS)
        print("   Chargement LAMNr...")
        Z_lam = load_z_lamnr(LAMNR_MODEL)
    except Exception as e:
        print(f"[ERREUR] {e}")
        return

    # Alignement
    n_min = min(len(df_target), len(Z_lin), len(Z_lam))
    df_target = df_target.iloc[:n_min]
    Z_lin = Z_lin[:n_min]
    Z_lam = Z_lam[:n_min]
    
    # --- ANALYSE joinedDX ---
    target_name = "joinedDX"
    
    # On filtre les NaNs
    temp_df = df_target[NUISANCE_COLS + [target_name]].copy()
    valid_mask = temp_df.notna().all(axis=1)
    
    # On ne garde que les classes avec assez de sujets (>= 40)
    # Sinon la CV plante ou n'a pas de sens
    y_all = temp_df.loc[valid_mask, target_name]
    counts = y_all.value_counts()
    valid_classes = counts[counts >= 40].index.tolist()
    
    print(f"   Classes conservées (N>=40) : {valid_classes}")
    
    # Filtre final
    valid_mask = valid_mask & temp_df[target_name].isin(valid_classes)
    n_valid = valid_mask.sum()
    print(f"   Sujets valides : {n_valid}")
    
    y = temp_df.loc[valid_mask, target_name].values
    X_nuis = temp_df.loc[valid_mask, NUISANCE_COLS].values
    Z_lin_sub = Z_lin[valid_mask]
    Z_lam_sub = Z_lam[valid_mask]
    
    # Encodage (String -> Int)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print(f"\n--- Résultats Classification ({len(valid_classes)} classes) ---")
    print(f"{'Modèle':<15} | {'Accuracy':<10} | {'AUC (Macro)':<10}")
    print("-" * 45)
    
    # 1. Nuisance Only
    acc_nuis, auc_nuis = evaluate_classification(X_nuis, y_enc)
    print(f"{'Nuisance':<15} | {acc_nuis:.3f}      | {auc_nuis:.3f}")
    
    # 2. Linear (SiMLR)
    acc_lin, auc_lin = evaluate_classification(np.hstack([X_nuis, Z_lin_sub]), y_enc)
    print(f"{'Linear':<15} | {acc_lin:.3f}      | {auc_lin:.3f}")
    
    # 3. LAMNr
    acc_lam, auc_lam = evaluate_classification(np.hstack([X_nuis, Z_lam_sub]), y_enc)
    print(f"{'LAMNr':<15} | {acc_lam:.3f}      | {auc_lam:.3f}")
    
    print("-" * 45)
    print(f"Uplift (Acc) : {acc_lam - acc_lin:+.3f}")
    print(f"Uplift (AUC) : {auc_lam - auc_lin:+.3f}")

if __name__ == "__main__":
    run_analysis()