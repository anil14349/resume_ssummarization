"""Train and export ML cleanup model."""
import logging
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from datetime import datetime
import shutil

from models.ml_cleanup import MLCleanupEnhancer
from prepare_cleanup_data import CleanupDataPreparator
from visualization import MetricsVisualizer

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        checkpoint_dir: str = "checkpoints"
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in monitored value
            checkpoint_dir: Directory to save checkpoints
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss: float, model: MLCleanupEnhancer, epoch: int):
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save checkpoint for
            epoch: Current epoch number
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0
            
    def save_checkpoint(self, val_loss: float, model: MLCleanupEnhancer, epoch: int):
        """Save model checkpoint.
        
        Args:
            val_loss: Validation loss for checkpoint
            model: Model to save
            epoch: Current epoch number
        """
        if val_loss < self.val_loss_min:
            logger.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})")
            self.val_loss_min = val_loss
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}"
            model.save(str(checkpoint_path))
            
            # Save metadata
            metadata = {
                "epoch": epoch,
                "val_loss": val_loss,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(checkpoint_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
    def load_best_checkpoint(self) -> MLCleanupEnhancer:
        """Load the best checkpoint based on validation loss."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*"))
        best_checkpoint = None
        best_loss = float('inf')
        
        for checkpoint in checkpoints:
            metadata_file = checkpoint / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    if metadata["val_loss"] < best_loss:
                        best_loss = metadata["val_loss"]
                        best_checkpoint = checkpoint
                        
        if best_checkpoint:
            logger.info(f"Loading best checkpoint from {best_checkpoint}")
            return MLCleanupEnhancer.load(str(best_checkpoint))
        else:
            raise ValueError("No checkpoints found")

class TrainingLogger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: str):
        """Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.csv"
        self.model_performance_file = self.log_dir / f"model_performance_{timestamp}.txt"
        
        # Initialize CSV writer
        self.metrics_file.write_text("epoch,split,metric,value\n")
        
    def log_metrics(self, epoch: int, metrics: dict, split: str = "train"):
        """Log metrics to CSV file."""
        with open(self.metrics_file, "a") as f:
            writer = csv.writer(f)
            for metric, value in metrics.items():
                writer.writerow([epoch, split, metric, value])
                
    def log_performance(self, message: str):
        """Log model performance message."""
        with open(self.model_performance_file, "a") as f:
            f.write(f"{datetime.now()}: {message}\n")

def train_cleanup_model(
    train_examples,
    val_examples,
    output_dir: str = "models/cleanup",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    patience: int = 5,
    checkpoint_dir: str = "checkpoints/cleanup"
):
    """Train cleanup model with early stopping and checkpointing.
    
    Args:
        train_examples: List of (raw, clean) training pairs
        val_examples: List of (raw, clean) validation pairs
        output_dir: Directory to save final model
        num_epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        patience: Patience for early stopping
        checkpoint_dir: Directory to save checkpoints
    """
    try:
        # Initialize model, logger, and early stopping
        cleanup = MLCleanupEnhancer()
        logger = TrainingLogger("logs/cleanup")
        early_stopping = EarlyStopping(
            patience=patience,
            checkpoint_dir=checkpoint_dir
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            cleanup.cleanup_model.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        logger.log_performance("Starting training")
        
        for epoch in range(num_epochs):
            # Training
            cleanup.cleanup_model.train()
            train_losses = []
            train_metrics = {
                'bleu': [], 'meteor': [], 'rouge1_f': [],
                'rouge2_f': [], 'rougeL_f': [], 'bert_score': []
            }
            
            for batch_idx in tqdm(range(0, len(train_examples), batch_size)):
                batch = train_examples[batch_idx:batch_idx + batch_size]
                raw_texts = [x[0] for x in batch]
                clean_texts = [x[1] for x in batch]
                
                # Forward pass
                loss = 0
                for raw, clean in zip(raw_texts, clean_texts):
                    output = cleanup.clean_output(raw)
                    metrics = CleanupDataPreparator.evaluate_summary(output, clean)
                    
                    # Accumulate metrics
                    for k, v in metrics.items():
                        train_metrics[k].append(v)
                        
                    # Calculate loss
                    loss += 1.0 - metrics['bert_score']  # Use BERTScore as loss
                    
                loss = loss / len(batch)
                train_losses.append(loss)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            # Validation
            cleanup.cleanup_model.eval()
            val_losses = []
            val_metrics = {
                'bleu': [], 'meteor': [], 'rouge1_f': [],
                'rouge2_f': [], 'rougeL_f': [], 'bert_score': []
            }
            
            with torch.no_grad():
                for batch_idx in range(0, len(val_examples), batch_size):
                    batch = val_examples[batch_idx:batch_idx + batch_size]
                    raw_texts = [x[0] for x in batch]
                    clean_texts = [x[1] for x in batch]
                    
                    for raw, clean in zip(raw_texts, clean_texts):
                        output = cleanup.clean_output(raw)
                        metrics = CleanupDataPreparator.evaluate_summary(output, clean)
                        
                        # Accumulate metrics
                        for k, v in metrics.items():
                            val_metrics[k].append(v)
                            
                        # Calculate loss
                        loss = 1.0 - metrics['bert_score']
                        val_losses.append(loss)
                        
            # Calculate epoch metrics
            train_loss = sum(train_losses) / len(train_losses)
            val_loss = sum(val_losses) / len(val_losses)
            
            train_epoch_metrics = {
                k: sum(v) / len(v) for k, v in train_metrics.items()
            }
            val_epoch_metrics = {
                k: sum(v) / len(v) for k, v in val_metrics.items()
            }
            
            # Log metrics
            train_epoch_metrics['loss'] = train_loss
            val_epoch_metrics['loss'] = val_loss
            
            logger.log_metrics(epoch + 1, train_epoch_metrics, "train")
            logger.log_metrics(epoch + 1, val_epoch_metrics, "val")
            
            # Log performance
            performance_msg = (
                f"Epoch {epoch + 1}/{num_epochs}\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Train Metrics: {train_epoch_metrics}\n"
                f"Val Metrics: {val_epoch_metrics}\n"
            )
            logger.log_performance(performance_msg)
            
            # Early stopping check
            early_stopping(val_loss, cleanup, epoch)
            
            if early_stopping.early_stop:
                logger.log_performance(
                    f"Early stopping triggered at epoch {epoch + 1}"
                )
                break
                
        # Load best model
        cleanup = early_stopping.load_best_checkpoint()
        
        # Save final model
        output_path = Path(output_dir)
        cleanup.save(str(output_path))
        logger.log_performance(f"Saved final model to {output_path}")
        
        # Generate training report
        metrics_file = sorted(Path("logs/cleanup").glob("metrics_*.csv"))[-1]
        visualizer = MetricsVisualizer(str(metrics_file))
        visualizer.generate_report("training_report.pdf")
        
        return cleanup
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def export_model(model_dir: str, output_dir: str):
    """Export trained model for deployment.
    
    Args:
        model_dir: Directory containing trained model
        output_dir: Directory to save exported model
    """
    try:
        # Load best model
        cleanup = MLCleanupEnhancer.load(model_dir)
        
        # Save in deployment format
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        cleanup.save(str(output_path))
        
        # Save config
        config = {
            "model_type": "ml_cleanup",
            "version": "1.0",
            "components": {
                "quality_model": "quality",
                "cleanup_model": "cleanup",
                "style_model": "style"
            }
        }
        
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Exported model to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load prepared data
    data_dir = Path("data/cleanup")
    with open(data_dir / "train_examples.json") as f:
        train_examples = json.load(f)["examples"]
    with open(data_dir / "val_examples.json") as f:
        val_examples = json.load(f)["examples"]
        
    # Train model with early stopping
    trained_model = train_cleanup_model(
        train_examples,
        val_examples,
        output_dir="models/cleanup",
        num_epochs=20,  # Increased epochs since we have early stopping
        patience=5
    )
    
    # Export model
    export_model(
        model_dir="models/cleanup",
        output_dir="exported_models/cleanup"
    )
