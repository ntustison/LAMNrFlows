import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Importation de votre nouveau trainer (ajustez le chemin selon votre structure)
from lamnrflows.training.train_lamnr_flows_tabular_v2 import TabularLAMNrTrainer

# ---------------------------------------------------------
# 1. Faux Modèle (Mock Model) pour contourner Glow/RealNVP
# ---------------------------------------------------------
class DummyFlowModel(nn.Module):
    """Un faux flux bijectif ultra-léger pour les tests unitaires."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward_and_log_det(self, z):
        # Fausse passe directe (recon) : multiplie par 2
        x_rec = z * 2.0
        log_det = torch.zeros(z.shape[0], device=z.device)
        return x_rec, log_det

    def inverse_and_log_det(self, x):
        # Fausse passe inverse (encodage) : divise par 2
        z = x * 0.5
        log_det = torch.zeros(x.shape[0], device=x.device)
        return z, log_det

# ---------------------------------------------------------
# 2. Fixtures Pytest (Préparation des données/arguments)
# ---------------------------------------------------------
@pytest.fixture
def dummy_args(tmp_path):
    """Génère de faux arguments CLI en utilisant le dossier temporaire de pytest."""
    class Args:
        pass
    
    args = Args()
    args.num_views = 2
    args.view = ["clinique_t1", "clinique_t2"]
    args.out_dir = str(tmp_path / "runs_test")
    
    # Arguments pour l'exportation que nous venons de créer
    args.save_z = True
    args.save_whitened = True
    args.save_recon = True
    
    # Arguments requis par BaseLAMNrTrainer
    args.align = "vicreg"
    args.align_weight = 1.0
    args.weighting = "fixed"
    args.device = "cpu"  # Force le CPU pour les tests CI
    
    return args

@pytest.fixture
def dummy_tabular_trainer(dummy_args):
    """Instancie le trainer en contournant le chargement réel des données."""
    # Instanciation (on suppose que __init__ accepte args)
    # Si votre __init__ charge les données immédiatement, nous allons
    # mocker la méthode build_loaders ou injecter un faux dataloader.
    trainer = TabularLAMNrTrainer(dummy_args)
    
    # Injection de faux modèles
    trainer.models = nn.ModuleList([DummyFlowModel(dim=10), DummyFlowModel(dim=10)])
    trainer.dev = torch.device("cpu")
    
    # Injection d'un faux DataLoader (1 batch, 2 vues, taille du batch=5, features=10)
    faux_batch = [torch.randn(5, 10), torch.randn(5, 10)]
    trainer.train_loader = [faux_batch]  # Une liste agit comme un itérable d'1 itération
    
    # Faux normaliseurs pour tester la branche "whitened"
    class DummyNormalizer:
        def normalize(self, z):
            return z - 0.1 # Fausse standardisation
            
    trainer.dataset_normalizers = [DummyNormalizer(), DummyNormalizer()]
    
    return trainer

# ---------------------------------------------------------
# 3. Les Tests Unitaires
# ---------------------------------------------------------
def test_tabular_trainer_inheritance(dummy_tabular_trainer):
    """Vérifie que le TabularTrainer hérite bien de la base unifiée."""
    from lamnrflows.training.base_trainer import BaseLAMNrTrainer
    assert isinstance(dummy_tabular_trainer, BaseLAMNrTrainer), "TabularLAMNrTrainer doit hériter de BaseLAMNrTrainer"

def test_export_tabular_results(dummy_tabular_trainer, dummy_args):
    """Vérifie que la nouvelle fonction d'exportation génère bien les bons fichiers CSV."""
    # 1. Exécution de la fonction d'exportation
    dummy_tabular_trainer.export_tabular_results()
    
    out_dir = Path(dummy_args.out_dir)
    
    # 2. Assertions : Vérifier que les fichiers existent sur le disque
    for view_name in dummy_args.view:
        path_z = out_dir / f"{view_name}_latent_z.csv"
        path_w = out_dir / f"{view_name}_whitened_epsilon.csv"
        path_r = out_dir / f"{view_name}_reconstructed_x.csv"
        
        assert path_z.exists(), f"Le fichier Z {path_z} n'a pas été créé."
        assert path_w.exists(), f"Le fichier Whitened {path_w} n'a pas été créé."
        assert path_r.exists(), f"Le fichier Recon {path_r} n'a pas été créé."
        
        # 3. Assertions : Vérifier le contenu (Optionnel mais recommandé)
        df_z = pd.read_csv(path_z)
        assert df_z.shape == (5, 10), "La forme du DataFrame Z exporté est incorrecte."