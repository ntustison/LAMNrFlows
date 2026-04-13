import pandas as pd
from pathlib import Path
import sys

# --- CONFIGURATION ---
DATA_DIR = Path("lamnr_repro_pack/processed/trimmed_input")
RAW_DIR = Path("lamnr_repro_pack/raw_data")

# 1. Fichiers de Référence (Images alignées qui ont encore l'ID)
# Utilisez les fichiers "aligned_" (pas "clean_aligned_" qui n'ont plus d'ID)
REF_FILES = {
    "NNL": DATA_DIR / "input_NNL_T1.csv",          # Pour NNL, l'input original a souvent l'ID
    "PPMI": DATA_DIR / "input_PPMI_T1.csv" # Pour PPMI, celui généré par master_clean_ppmi.py (avant suppression ID)
    # Note : Si master_clean_ppmi.py a écrasé le fichier sans garder l'ID,
    # il faudra peut-être utiliser le fichier "aligned_input_PPMI_T1.csv" de l'étape intermédiaire.
    # Si vous n'avez que les "clean_", ce script ne peut pas fonctionner car il n'y a plus de clé de jointure.
}

# 2. Fichiers Démographiques Cibles
DEMO_FILES = {
    "NNL": RAW_DIR / "nnl_demo_and_targets.csv",
    "PPMI": RAW_DIR / "ppmi_demo_and_targets.csv" # Ajustez le nom ici
}

def find_id_column(df, filename):
    candidates = ["subjectid", "patno", "oid", "id", "index", "participant_id"]
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return df.columns[0]

def align_demographics():
    print("--- Alignement Démographiques v2 (Avec Dédoublonnage) ---")
    
    for dataset, ref_path in REF_FILES.items():
        if dataset not in DEMO_FILES: continue
        demo_path = DEMO_FILES[dataset]
        
        print(f"\nTraitement : {dataset}")
        
        # A. Vérifications
        if not ref_path.exists():
            print(f"[ERREUR] Réf introuvable : {ref_path}")
            continue
        if not demo_path.exists():
            print(f"[ERREUR] Demo introuvable : {demo_path}")
            continue

        # B. Chargement Référence (La vérité terrain)
        # On lit la 1ère colonne qui est supposée être l'ID
        df_ref = pd.read_csv(ref_path)
        ref_id_col = df_ref.columns[0]
        
        # IDs de référence (déjà uniques grâce à master_clean_ppmi)
        ref_ids = df_ref[ref_id_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).unique()
        print(f"   Sujets Images (Ref) : {len(ref_ids)}")

        # C. Chargement Démographie
        df_demo = pd.read_csv(demo_path, low_memory=False)
        demo_id_col = find_id_column(df_demo, demo_path.name)
        print(f"   Clé Demo identifiée : '{demo_id_col}'")
        
        # Standardisation ID
        df_demo["_ALIGN_KEY"] = df_demo[demo_id_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        
        # --- DÉDOUBLONNAGE EXPLICITE ---
        n_orig = len(df_demo)
        # On garde la première occurrence (keep='first')
        df_demo = df_demo.drop_duplicates(subset="_ALIGN_KEY", keep='first')
        n_dedup = len(df_demo)
        
        if n_orig != n_dedup:
            print(f"   [NETTOYAGE] {n_orig - n_dedup} doublons supprimés dans la démo.")
        
        # D. Alignement Strict
        # 1. On ne garde que les sujets présents dans l'image
        df_aligned = df_demo[df_demo["_ALIGN_KEY"].isin(ref_ids)].copy()
        
        # 2. On impose l'ordre des images (Ref)
        df_aligned = df_aligned.set_index("_ALIGN_KEY").reindex(ref_ids)
        
        # 3. Reset index
        df_aligned = df_aligned.reset_index().rename(columns={"index": demo_id_col})
        
        # E. Sauvegarde
        out_name = f"aligned_targets_{dataset}.csv"
        out_path = DATA_DIR / out_name
        df_aligned.to_csv(out_path, index=False)
        
        # Vérification finale des dimensions
        print(f"   -> Sauvegardé : {out_name}")
        print(f"      Lignes : {len(df_aligned)} (Doit être égal à {len(ref_ids)})")

if __name__ == "__main__":
    align_demographics()