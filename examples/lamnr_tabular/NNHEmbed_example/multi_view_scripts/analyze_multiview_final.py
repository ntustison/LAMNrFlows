import json
import pandas as pd
from pathlib import Path
import re
import sys

# --- CONFIGURATION ---
RUNS_DIR = Path("runs/multiview_production")

def parse_filename(filename):
    """Extrait la méthode et le lambda du nom de fichier."""
    # Ex: vicreg_lambda1.0_metrics.json -> method=vicreg, lambda=1.0
    # Ex: seed42_baseline_metrics.json -> method=baseline, lambda=0
    
    name = filename.name
    if "baseline" in name:
        return "Baseline", 0.0
    
    # Regex pour les méthodes avec lambda
    match = re.search(r"([a-z_]+)_lambda([0-9\.]+)_metrics", name)
    if match:
        return match.group(1), float(match.group(2))
    
    return "Unknown", 0.0

def analyze_multiview():
    print(f"--- Analyse Finale Multi-Vues : {RUNS_DIR} ---")
    
    if not RUNS_DIR.exists():
        print(f"[ERREUR] Dossier introuvable.")
        sys.exit(1)

    records = []
    
    # Parcours des datasets (NNL, PPMI)
    for dataset_dir in RUNS_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        json_files = list(dataset_dir.glob("*_metrics.json"))
        
        for f in json_files:
            method, lam = parse_filename(f)
            
            try:
                with open(f, "r") as json_file:
                    data = json.load(json_file)
                    
                    # Extraction robustes des métriques
                    metrics = data.get("metrics", {})
                    
                    # Score principal (BPD) - Plus bas est mieux
                    val_bpd = metrics.get("best_val_bpd")
                    if val_bpd is None:
                         val_bpd = data.get("val_bpd") # Format alternatif
                    
                    # Si dispo : Loss totale (BPD + Pénalité)
                    total_loss = metrics.get("best_metric")
                    
                    if val_bpd is not None:
                        records.append({
                            "Dataset": dataset_name,
                            "Method": method,
                            "Lambda": lam,
                            "Val BPD": float(val_bpd),
                            "Total Loss": float(total_loss) if total_loss else None,
                            "File": f.name
                        })
            except Exception as e:
                print(f"Erreur lecture {f.name}: {e}")

    if not records:
        print("Aucune donnée trouvée.")
        return

    # --- SYNTHÈSE ---
    df = pd.DataFrame(records)
    
    # Tri pour l'affichage
    df = df.sort_values(by=["Dataset", "Method", "Lambda"])
    
    print("\n=== RÉSULTATS DÉTAILLÉS ===")
    print(df[["Dataset", "Method", "Lambda", "Val BPD"]].to_string(index=False))
    
    # --- LES CHAMPIONS ---
    print("\n\n=== MEILLEURE CONFIGURATION PAR MÉTHODE (BPD le plus bas) ===")
    
    summary = df.loc[df.groupby(["Dataset", "Method"])["Val BPD"].idxmin()]
    summary = summary[["Dataset", "Method", "Lambda", "Val BPD"]].sort_values(by=["Dataset", "Val BPD"])
    
    print(summary.to_string(index=False))

    # --- LE GRAND GAGNANT PAR DATASET ---
    print("\n\n=== LE GRAND GAGNANT (Overall Best) ===")
    best_overall = df.loc[df.groupby("Dataset")["Val BPD"].idxmin()]
    print(best_overall[["Dataset", "Method", "Lambda", "Val BPD"]].to_string(index=False))
    
    # Export CSV
    df.to_csv("synthese_multiview_results.csv", index=False)
    print("\nSauvegardé dans 'synthese_multiview_results.csv'")

if __name__ == "__main__":
    analyze_multiview()