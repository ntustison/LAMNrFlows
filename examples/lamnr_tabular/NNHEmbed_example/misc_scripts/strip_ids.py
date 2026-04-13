import pandas as pd
from pathlib import Path
import sys

# --- CONFIGURATION ---
DATA_DIR = Path("lamnr_repro_pack/processed/trimmed_input")

# Liste de toutes les vues à nettoyer
# Pour PPMI : on prend les "aligned"
# Pour NNL : on prend les originaux (ou aligned s'ils existent déjà)
FILES_TO_CLEAN = {
    "PPMI_T1": "input_PPMI_T1.csv",
    "PPMI_DTI": "input_PPMI_DTI.csv",
    "PPMI_rsfMRI": "input_PPMI_rsfMRI.csv",
    "NNL_T1": "input_NNL_T1.csv",
    "NNL_DTI": "input_NNL_DTI.csv",
    "NNL_rsfMRI": "input_NNL_rsfMRI.csv"
}

def clean_features():
    print("--- Préparation des Features Finales (Suppression Index/IDs) ---")
    
    for label, filename in FILES_TO_CLEAN.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"[SKIP] {filename} introuvable.")
            continue
            
        print(f"\nTraitement : {filename}")
        df = pd.read_csv(path)
        
        # --- LOGIQUE DE SUPPRESSION ---
        cols_to_drop = []
        
        # 1. Identifier la première colonne (souvent l'index ou ID)
        first_col = df.columns[0]
        
        # Liste des noms suspects pour un ID
        suspect_names = ["index", "subjectid", "patno", "id", "run_id", "unnamed: 0", "oid"]
        
        # Si la première colonne est un ID ou un index explicite
        if first_col.lower() in suspect_names or "id" in first_col.lower():
            cols_to_drop.append(first_col)
        # Si la première colonne contient des chaînes de caractères (ex: "3001" ou "sub-01")
        elif df[first_col].dtype == 'object':
             cols_to_drop.append(first_col)
             
        # On regarde aussi si d'autres colonnes s'appellent "index"
        for c in df.columns:
            if c.lower() == "index" and c not in cols_to_drop:
                cols_to_drop.append(c)

        # --- ACTION ---
        if cols_to_drop:
            print(f"   Suppression de : {cols_to_drop}")
            df_clean = df.drop(columns=cols_to_drop)
        else:
            print("   [ATTENTION] Aucune colonne ID détectée automatiquement.")
            print(f"   Première colonne : {first_col} (Type: {df[first_col].dtype})")
            # Sécurité : Si c'est un entier qui ressemble à un index (0, 1, 2...), on demande confirmation ou on drop
            # Ici, on assume que vous voulez forcer le drop de la 1ere colonne par sécurité
            print("   -> Force suppression colonne 0 par sécurité.")
            df_clean = df.iloc[:, 1:] 

        # Vérification finale
        print(f"   Dimensions : {df.shape} -> {df_clean.shape}")
        
        # Sauvegarde avec préfixe "clean_"
        out_name = f"clean_{filename}"
        out_path = DATA_DIR / out_name
        df_clean.to_csv(out_path, index=False)
        print(f"   -> Sauvegardé : {out_name}")

if __name__ == "__main__":
    clean_features()