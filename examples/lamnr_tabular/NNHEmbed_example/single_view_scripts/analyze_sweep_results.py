import json
import pandas as pd
from pathlib import Path
import re
import sys

# --- CONFIGURATION ---
RUNS_DIR = Path("runs/validation_targeted")

def analyze_results():
    print(f"--- Analyse des Résultats (v2) : {RUNS_DIR} ---")
    
    if not RUNS_DIR.exists():
        print(f"[ERREUR] Le dossier {RUNS_DIR} n'existe pas.")
        return

    # 1. Collecte des données
    records = []
    metric_files = list(RUNS_DIR.rglob("out_metrics.json"))
    
    print(f"Fichiers métriques trouvés : {len(metric_files)}")
    
    for f in metric_files:
        # Structure du chemin: .../NomDataset/seedXX_KXX_hcXX/out_metrics.json
        run_dir = f.parent
        dataset_name = run_dir.parent.name
        run_name = run_dir.name
        
        # Extraction K et HC via Regex
        match = re.search(r"K(\d+)_hc(\d+)", run_name)
        if not match:
            continue
            
        k_val = int(match.group(1))
        hc_val = int(match.group(2))
        
        # Lecture du JSON
        try:
            with open(f, "r") as json_file:
                data = json.load(json_file)
                
                # --- CORRECTION ICI : Accès au sous-dictionnaire 'metrics' ---
                metrics_dict = data.get("metrics", {})
                
                # On cherche 'best_val_bpd' ou 'best_metric'
                val_score = metrics_dict.get("best_val_bpd")
                if val_score is None:
                    val_score = metrics_dict.get("best_metric")
                
                # Fallback: si pas dans metrics, on regarde à la racine (vieux format)
                if val_score is None:
                    val_score = data.get("val_bpd")

                if val_score is not None:
                    records.append({
                        "Dataset": dataset_name,
                        "K": k_val,
                        "HC": hc_val,
                        "Score": float(val_score), # BPD (Lower is better)
                        "Path": str(run_dir)
                    })
        except Exception as e:
            print(f"Erreur lecture {f}: {e}")

    if not records:
        print("Aucune donnée valide trouvée (vérifiez les fichiers JSON).")
        return

    # 2. Agrégation
    df = pd.DataFrame(records)
    
    # Moyenne sur les seeds
    summary = df.groupby(["Dataset", "K", "HC"])["Score"].agg(["mean", "std", "count"]).reset_index()
    
    # 3. Affichage des Gagnants
    print("\n=== RÉSULTATS : MEILLEURE ARCHITECTURE PAR DATASET ===")
    print("(Critère : 'val_bpd', plus bas est mieux)\n")
    
    unique_datasets = summary["Dataset"].unique()
    
    for ds in unique_datasets:
        ds_data = summary[summary["Dataset"] == ds]
        
        # On cherche le score MINIMUM
        best_idx = ds_data["mean"].idxmin()
        best_row = ds_data.loc[best_idx]
        
        print(f">> {ds}")
        print(f"   GAGNANT : K={int(best_row['K'])}, HC={int(best_row['HC'])}")
        print(f"   Score   : {best_row['mean']:.4f} (+/- {best_row['std']:.4f})")
        
        # Comparaison avec l'hypothèse (K=4, HC=80)
        hypo = ds_data[(ds_data["K"] == 4) & (ds_data["HC"] == 80)]
        if not hypo.empty:
            hypo_score = hypo.iloc[0]["mean"]
            diff = hypo_score - best_row["mean"]
            # Si diff est positif, c'est que l'hypothèse est pire (score plus haut)
            status = "EXACT" if diff == 0 else ("Pire" if diff > 0 else "Mieux")
            print(f"   (vs K=4,HC=80 : {hypo_score:.4f} | Diff: {diff:+.4f} -> {status})")
        else:
            print("   (Pas de données pour K=4, HC=80)")
        print("-" * 40)

    # 4. Export
    summary.to_csv("synthese_hyperparametres.csv", index=False)
    print("\nTableau complet sauvegardé dans 'synthese_hyperparametres.csv'")

if __name__ == "__main__":
    analyze_results()