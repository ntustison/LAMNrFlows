import pandas as pd
from pathlib import Path
import sys

# --- CONFIGURATION ---
# Dossier contenant vos fichiers bruts
DATA_DIR = Path("lamnr_repro_pack/processed/latent_projections/")

# Noms des fichiers sources
FILES = {
    "T1": "projection_NNL_T1.csv",
    "DTI": "projection_NNL_DTI.csv",
    "rsfMRI": "projection_NNL_rsfMRI.csv"
}

def clean_and_coordinate():
    print("--- Nettoyage Maître NNL (Dédoublonnage + Intersection) ---")
    
    if not DATA_DIR.exists():
        print(f"[ERREUR] Dossier {DATA_DIR} introuvable.")
        sys.exit(1)

    # 1. Lecture et Dédoublonnage
    dfs = {}
    
    for modality, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"[ERREUR] Fichier manquant : {filename}")
            sys.exit(1)
            
        print(f"\nLecture : {filename}")
        df = pd.read_csv(path, low_memory=False)
        
        # Identification de la 1ère colonne (ID / Index)
        id_col = df.columns[0]
        print(f"   Clé (1ère colonne) : '{id_col}'")
        
        # Nettoyage des IDs (String + Strip) pour éviter les faux doublons (ex: "123 " vs "123")
        # On travaille sur une copie pour ne pas casser l'original tout de suite
        ids_clean = df[id_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        df["_TEMP_ID"] = ids_clean
        
        # Dédoublonnage sur la 1ère colonne
        # keep='first' garde la première occurrence rencontrée
        n_before = len(df)
        df = df.drop_duplicates(subset="_TEMP_ID", keep='first')
        n_after = len(df)
        
        if n_before != n_after:
            print(f"   -> Doublons retirés : {n_before - n_after}")
        else:
            print(f"   -> Aucun doublon.")
            
        dfs[modality] = df

    # 2. Coordination (Intersection)
    print(f"\n--- Coordination des Vues ---")
    # On récupère les sets d'IDs de chaque vue
    id_sets = [set(df["_TEMP_ID"]) for df in dfs.values()]
    
    # Intersection : on ne garde que les IDs présents PARTOUT
    common_ids = set.intersection(*id_sets)
    print(f"Sujets communs aux 3 vues : {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("[CRITIQUE] Intersection vide ! Vérifiez vos fichiers.")
        sys.exit(1)
        
    # Liste triée pour garantir l'ordre (alignement ligne à ligne)
    sorted_common_ids = sorted(list(common_ids))

    # 3. Filtrage et Sauvegarde
    print(f"\n--- Sauvegarde (Prêt pour Entraînement) ---")
    
    for modality, filename in FILES.items():
        df = dfs[modality]
        
        # A. On ne garde que les communs
        df_final = df[df["_TEMP_ID"].isin(common_ids)].copy()
        
        # B. On trie par ID (Crucial pour l'alignement)
        df_final = df_final.set_index("_TEMP_ID").reindex(sorted_common_ids)
        
        # C. On retire la première colonne (l'ID original) et la colonne temporaire
        # Le trainer ne doit voir QUE des chiffres (les features)
        # On drop l'ID original (colonne 0)
        df_final = df_final.drop(columns=[df_final.columns[0]])
        
        # Nom du fichier de sortie
        out_name = f"clean_aligned_{filename}"
        out_path = DATA_DIR / out_name
        
        df_final.to_csv(out_path, index=False)
        print(f"   -> {out_name} : {df_final.shape} (Sans colonne ID)")

if __name__ == "__main__":
    clean_and_coordinate()