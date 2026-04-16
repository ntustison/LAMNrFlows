from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


def _save_metric_plots(csv_path: Path, out_dir: Path, remove_spikes: bool = False):
    if not csv_path.exists():
        return
    iters, losses, bpds = [], [], []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                # Le bloc try/except évite un crash si le script lit la ligne exactement 
                # au moment où l'écrivain CSV est en train de l'enregistrer.
                try:
                    it, loss, bpd = int(float(row[0])), float(row[1]), float(row[2])
                    iters.append(it); losses.append(loss); bpds.append(bpd)
                except ValueError:
                    continue
                    
        if len(iters) < 2:
            return
            
        if remove_spikes and len(losses) > 10:
            s_losses = pd.Series(losses)
            
            # Fenêtre adaptative (max 50, s'ajuste si le fichier est petit)
            w = min(50, max(5, len(losses) // 10))
            
            # Calcul de la médiane glissante locale
            rolling_med = s_losses.rolling(window=w, center=True, min_periods=1).median()
            
            # Calcul de l'écart absolu de chaque point par rapport à la médiane
            diff = np.abs(s_losses - rolling_med)
            
            # Calcul du MAD (Median Absolute Deviation) local
            rolling_mad = diff.rolling(window=w, center=True, min_periods=1).median()
            
            # Un 'spike' est un point qui dévie de plus de 5 fois le MAD (avec une tolérance minimale de sécurité)
            is_spike = diff > (5 * rolling_mad + 1e-6)
            
            # Remplacer les valeurs par NaN pour créer une cassure visuelle sur le graphique
            losses = np.where(is_spike, np.nan, losses)
            bpds = np.where(is_spike, np.nan, bpds)

        plt.figure()
        plt.plot(iters, losses)
        plt.xlabel("iter"); plt.ylabel("loss"); plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png"); plt.close()
        
        plt.figure()
        plt.plot(iters, bpds)
        plt.xlabel("iter"); plt.ylabel("sum_bpd"); plt.title("Sum BPD (training batches)")
        plt.tight_layout()
        plt.savefig(out_dir / "bpd_curve.png"); plt.close()
        
    except Exception as e:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save training metric plots from CSV logs.")
    parser.add_argument("--csv_path", type=Path, required=True, help="Path to the CSV log file.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to save the plots.")
    parser.add_argument("--remove_spikes", action="store_true", help="Remove spikes from the plots.")
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _save_metric_plots(args.csv_path, args.out_dir, args.remove_spikes)