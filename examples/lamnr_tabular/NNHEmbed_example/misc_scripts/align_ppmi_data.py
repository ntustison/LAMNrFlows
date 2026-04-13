import pandas as pd
from pathlib import Path
import sys

# --- CONFIGURATION ---
DATA_DIR = Path("lamnr_repro_pack/processed/trimmed_input")

FILES = {
    "T1": "input_PPMI_T1.csv",
    "DTI": "input_PPMI_DTI.csv",
    "rsfMRI": "input_PPMI_rsfMRI.csv"
}

def clean_and_align():
    print("--- Nettoyage (Doublons) et Alignement PPMI ---")
    
    if not DATA_DIR.exists():
        print(f"[ERREUR] Dossier introuvable : {DATA_DIR}")
        sys.exit(1)

    dfs = {}
    
    # 1. Chargement et Dédoublonnage
    for modality, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"[ERREUR] Fichier manquant : {path}")
            sys.exit(1)
            
        print(f"\nTraitement {modality} ({filename})...")
        df = pd.read_csv(path, low_memory=False)
        
        # Identification de la colonne ID
        # Le user a mentionné "index". On cherche 'index', 'subjectID', 'PATNO', ou la 1ère colonne.
        possible_ids = [c for c in df.columns if c.lower() in ["index", "subjectid", "patno", "id", "run_id"]]
        if possible_ids:
            id_col = possible_ids[0]
        else:
            id_col = df.columns[0]
            
        print(f"   Clé identifiée : '{id_col}'")
        
        # --- DÉDOUBLONNAGE ---
        n_orig = len(df)
        # On garde la PREMIÈRE occurrence (keep='first')
        df = df.drop_duplicates(subset=id_col, keep='first')
        n_dedup = len(df)
        
        if n_orig != n_dedup:
            print(f"   [NETTOYAGE] {n_orig - n_dedup} doublons supprimés (Reste: {n_dedup})")
        else:
            print(f"   (Aucun doublon trouvé)")
            
        # Création d'une colonne ID standardisée pour l'alignement
        df["_ALIGN_ID"] = df[id_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        dfs[modality] = df

    # 2. Intersection (Sujets communs aux 3)
    sets = [set(df["_ALIGN_ID"]) for df in dfs.values()]
    common_ids = set.intersection(*sets)
    
    print(f"\n--- Intersection ---")
    print(f"Sujets communs aux 3 modalités : {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("[CRITIQUE] Intersection vide ! Vérifiez que les IDs sont compatibles entre les fichiers.")
        sys.exit(1)

    # 3. Sauvegarde des fichiers alignés
    sorted_ids = sorted(list(common_ids))
    
    for modality, filename in FILES.items():
        df = dfs[modality]
        
        # Filtrer
        df_aligned = df[df["_ALIGN_ID"].isin(common_ids)].copy()
        
        # Trier (Strictement nécessaire pour le multi-view)
        df_aligned = df_aligned.sort_values(by="_ALIGN_ID")
        
        # Nettoyer la colonne temporaire
        df_aligned = df_aligned.drop(columns=["_ALIGN_ID"])
        
        # Sauvegarder
        # On écrase les précédents "aligned_" s'ils existaient
        new_filename = f"aligned_{filename}"
        out_path = DATA_DIR / new_filename
        
        df_aligned.to_csv(out_path, index=False)
        print(f"   -> Sauvegardé : {new_filename} ({len(df_aligned)} lignes)")

    print("\n[PRÊT] Vous pouvez relancer vos scripts 'run_partX_...sh'.")

if __name__ == "__main__":
    clean_and_align()