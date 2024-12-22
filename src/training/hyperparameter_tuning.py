"""Hyperparameter tuning for ML cleanup model."""
import logging
from pathlib import Path
import json
from typing import Dict, List, Any
import optuna
from optuna.trial import Trial
import numpy as np
from sklearn.model_selection import KFold
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from models.ml_cleanup import MLCleanupEnhancer
from prepare_cleanup_data import CleanupDataPreparator
from visualization import MetricsVisualizer

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Tune hyperparameters using Optuna."""
    
    def __init__(
        self,
        train_examples: List,
        val_examples: List,
        n_trials: int = 20,
        n_folds: int = 5,
        study_name: str = "cleanup_optimization"
    ):
        """Initialize tuner.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            n_trials: Number of optimization trials
            n_folds: Number of cross-validation folds
            study_name: Name of optimization study
        """
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.study_name = study_name
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Setup logging
        self.log_dir = Path("logs/hyperparameter_tuning")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def objective(self, trial: Trial) -> float:
        """Optimization objective function."""
        # Sample hyperparameters
        params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_epochs": trial.suggest_int("num_epochs", 5, 30),
            "patience": trial.suggest_int("patience", 3, 10),
            "min_delta": trial.suggest_loguniform("min_delta", 1e-5, 1e-3),
            "dropout": trial.suggest_uniform("dropout", 0.1, 0.5),
            "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000)
        }
        
        # Cross validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_examples)):
            # Split data
            fold_train = [self.train_examples[i] for i in train_idx]
            fold_val = [self.train_examples[i] for i in val_idx]
            
            # Initialize model
            model = MLCleanupEnhancer()
            
            # Train model
            val_loss = self.train_fold(
                model,
                fold_train,
                fold_val,
                params,
                trial,
                fold
            )
            
            cv_scores.append(val_loss)
            
            # Report intermediate value
            trial.report(val_loss, fold)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(cv_scores)
        
    def train_fold(
        self,
        model: MLCleanupEnhancer,
        train_data: List,
        val_data: List,
        params: Dict[str, Any],
        trial: Trial,
        fold: int
    ) -> float:
        """Train model on a single fold."""
        # Initialize optimizer with warmup
        optimizer = torch.optim.AdamW(
            model.cleanup_model.parameters(),
            lr=params["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=params["warmup_steps"]
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(params["num_epochs"]):
            # Training
            model.cleanup_model.train()
            train_losses = []
            
            for i in range(0, len(train_data), params["batch_size"]):
                batch = train_data[i:i + params["batch_size"]]
                loss = self.train_batch(model, batch, optimizer)
                train_losses.append(loss)
                
                if i // params["batch_size"] < params["warmup_steps"]:
                    scheduler.step()
                    
            # Validation
            model.cleanup_model.eval()
            val_losses = []
            
            with torch.no_grad():
                for i in range(0, len(val_data), params["batch_size"]):
                    batch = val_data[i:i + params["batch_size"]]
                    loss = self.validate_batch(model, batch)
                    val_losses.append(loss)
                    
            val_loss = np.mean(val_losses)
            
            # Early stopping
            if val_loss < best_val_loss - params["min_delta"]:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= params["patience"]:
                break
                
            # Report intermediate value
            trial.report(val_loss, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return best_val_loss
        
    def train_batch(
        self,
        model: MLCleanupEnhancer,
        batch: List,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train on a single batch."""
        optimizer.zero_grad()
        
        loss = 0
        for raw, clean in batch:
            output = model.clean_output(raw)
            metrics = CleanupDataPreparator.evaluate_summary(output, clean)
            loss += 1.0 - metrics['bert_score']
            
        loss = loss / len(batch)
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    def validate_batch(
        self,
        model: MLCleanupEnhancer,
        batch: List
    ) -> float:
        """Validate on a single batch."""
        loss = 0
        for raw, clean in batch:
            output = model.clean_output(raw)
            metrics = CleanupDataPreparator.evaluate_summary(output, clean)
            loss += 1.0 - metrics['bert_score']
            
        return loss / len(batch)
        
    def optimize(self):
        """Run hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save results
        self.save_results()
        
        return self.study.best_params
        
    def save_results(self):
        """Save optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.log_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        with open(results_dir / "best_params.json", "w") as f:
            json.dump(self.study.best_params, f, indent=2)
            
        # Save optimization history
        history = {
            "values": self.study.trials_dataframe()["value"].tolist(),
            "params": [
                trial.params for trial in self.study.trials
            ]
        }
        
        with open(results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
            
        # Plot optimization history
        self.plot_optimization_history(results_dir)
        self.plot_parameter_importances(results_dir)
        self.plot_parallel_coordinate(results_dir)
        
    def plot_optimization_history(self, results_dir: Path):
        """Plot optimization history."""
        plt.figure(figsize=(10, 6))
        
        # Plot optimization history
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        
        plt.title("Optimization History")
        plt.tight_layout()
        plt.savefig(results_dir / "optimization_history.png")
        plt.close()
        
    def plot_parameter_importances(self, results_dir: Path):
        """Plot parameter importances."""
        plt.figure(figsize=(10, 6))
        
        # Plot parameter importances
        optuna.visualization.matplotlib.plot_param_importances(self.study)
        
        plt.title("Parameter Importances")
        plt.tight_layout()
        plt.savefig(results_dir / "parameter_importances.png")
        plt.close()
        
    def plot_parallel_coordinate(self, results_dir: Path):
        """Plot parallel coordinate plot."""
        plt.figure(figsize=(15, 8))
        
        # Plot parallel coordinate
        optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        
        plt.title("Parallel Coordinate Plot")
        plt.tight_layout()
        plt.savefig(results_dir / "parallel_coordinate.png")
        plt.close()
        
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    data_dir = Path("data/cleanup")
    with open(data_dir / "train_examples.json") as f:
        train_examples = json.load(f)["examples"]
    with open(data_dir / "val_examples.json") as f:
        val_examples = json.load(f)["examples"]
        
    # Initialize tuner
    tuner = HyperparameterTuner(
        train_examples=train_examples,
        val_examples=val_examples,
        n_trials=20,
        n_folds=5
    )
    
    # Run optimization
    best_params = tuner.optimize()
