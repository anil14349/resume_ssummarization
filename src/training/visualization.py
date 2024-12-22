"""Visualization tools for training metrics."""
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class MetricsVisualizer:
    """Visualize training metrics."""
    
    def __init__(self, metrics_file: str):
        """Initialize visualizer.
        
        Args:
            metrics_file: Path to metrics CSV file
        """
        self.metrics_df = pd.read_csv(metrics_file)
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_metric_progress(
        self,
        metric: str,
        save: bool = True,
        show: bool = True,
        interactive: bool = False
    ):
        """Plot training and validation progress for a metric.
        
        Args:
            metric: Name of metric to plot
            save: Whether to save plot
            show: Whether to display plot
            interactive: Whether to use plotly for interactive plot
        """
        if interactive:
            fig = go.Figure()
            
            for split in ['train', 'val']:
                data = self.metrics_df[
                    (self.metrics_df['metric'] == metric) &
                    (self.metrics_df['split'] == split)
                ]
                
                fig.add_trace(go.Scatter(
                    x=data['epoch'],
                    y=data['value'],
                    name=split,
                    mode='lines+markers'
                ))
                
            fig.update_layout(
                title=f"{metric} Progress",
                xaxis_title="Epoch",
                yaxis_title="Value",
                hovermode='x unified'
            )
            
            if save:
                fig.write_html(self.output_dir / f"{metric}_progress_interactive.html")
            if show:
                fig.show()
        else:
            plt.figure(figsize=(10, 6))
            
            for split in ['train', 'val']:
                data = self.metrics_df[
                    (self.metrics_df['metric'] == metric) &
                    (self.metrics_df['split'] == split)
                ]
                plt.plot(data['epoch'], data['value'], label=split)
                
            plt.title(f"{metric} Progress")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            
            if save:
                plt.savefig(self.output_dir / f"{metric}_progress.png")
            if show:
                plt.show()
            plt.close()
            
    def plot_learning_curves(
        self,
        metrics: Optional[List[str]] = None,
        save: bool = True,
        show: bool = True
    ):
        """Plot learning curves with confidence intervals."""
        if metrics is None:
            metrics = self.metrics_df['metric'].unique()
            
        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2,
            figsize=(15, 5 * ((n_metrics + 1) // 2))
        )
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            for split in ['train', 'val']:
                data = self.metrics_df[
                    (self.metrics_df['metric'] == metric) &
                    (self.metrics_df['split'] == split)
                ]
                
                # Calculate mean and std
                grouped = data.groupby('epoch')['value']
                mean = grouped.mean()
                std = grouped.std()
                
                # Plot mean and confidence interval
                axes[i].plot(mean.index, mean.values, label=split)
                axes[i].fill_between(
                    mean.index,
                    mean.values - std.values,
                    mean.values + std.values,
                    alpha=0.2
                )
                
            axes[i].set_title(f"{metric} Learning Curve")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Value")
            axes[i].legend()
            axes[i].grid(True)
            
        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "learning_curves.png")
        if show:
            plt.show()
        plt.close()
        
    def plot_metric_relationships(
        self,
        split: str = "train",
        save: bool = True,
        show: bool = True
    ):
        """Plot relationships between metrics using scatter matrix."""
        # Pivot data
        pivot_df = self.metrics_df[
            self.metrics_df['split'] == split
        ].pivot(
            index='epoch',
            columns='metric',
            values='value'
        )
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            pivot_df,
            dimensions=pivot_df.columns,
            title=f"Metric Relationships ({split})"
        )
        
        fig.update_layout(
            width=1000,
            height=1000
        )
        
        if save:
            fig.write_html(self.output_dir / f"metric_relationships_{split}.html")
        if show:
            fig.show()
            
    def plot_metric_embedding(
        self,
        method: str = "tsne",
        split: str = "train",
        save: bool = True,
        show: bool = True
    ):
        """Plot 2D embedding of metric space."""
        # Pivot data
        pivot_df = self.metrics_df[
            self.metrics_df['split'] == split
        ].pivot(
            index='epoch',
            columns='metric',
            values='value'
        )
        
        # Calculate embedding
        if method == "tsne":
            embedding = TSNE(n_components=2, random_state=42)
        else:
            embedding = PCA(n_components=2, random_state=42)
            
        coords = embedding.fit_transform(pivot_df)
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers+text',
            text=pivot_df.index,
            hovertemplate="Epoch: %{text}<br>" +
                         "x: %{x:.2f}<br>" +
                         "y: %{y:.2f}"
        ))
        
        fig.update_layout(
            title=f"2D {method.upper()} Embedding of Metric Space ({split})",
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            width=800,
            height=600
        )
        
        if save:
            fig.write_html(self.output_dir / f"metric_embedding_{method}_{split}.html")
        if show:
            fig.show()
            
    def plot_metric_radar(
        self,
        epoch: Optional[int] = None,
        save: bool = True,
        show: bool = True
    ):
        """Plot radar chart of metrics for specific epoch."""
        if epoch is None:
            epoch = self.metrics_df['epoch'].max()
            
        # Get data for epoch
        epoch_data = self.metrics_df[
            self.metrics_df['epoch'] == epoch
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        for split in ['train', 'val']:
            values = epoch_data[
                epoch_data['split'] == split
            ].set_index('metric')['value']
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=values.index,
                fill='toself',
                name=split
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"Metric Radar Chart (Epoch {epoch})"
        )
        
        if save:
            fig.write_html(self.output_dir / f"metric_radar_epoch_{epoch}.html")
        if show:
            fig.show()
            
    def plot_all_metrics(
        self,
        metrics: Optional[List[str]] = None,
        save: bool = True,
        show: bool = True,
        interactive: bool = False
    ):
        """Plot all metrics progress."""
        if metrics is None:
            metrics = self.metrics_df['metric'].unique()
            
        for metric in metrics:
            self.plot_metric_progress(metric, save, show, interactive)
            
    def generate_report(self, output_file: str):
        """Generate comprehensive PDF report."""
        from fpdf import FPDF
        
        pdf = FPDF()
        
        # Title page
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 20, "Training Metrics Report", ln=True, align="C")
        
        # Summary statistics
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Summary Statistics", ln=True)
        
        for split in ['train', 'val']:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"\n{split.capitalize()} Metrics:", ln=True)
            
            stats = self.metrics_df[
                self.metrics_df['split'] == split
            ].groupby('metric')['value'].agg([
                'mean', 'std', 'min', 'max',
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75)
            ])
            
            stats.columns = ['Mean', 'Std', 'Min', 'Max', 'Q1', 'Q3']
            
            for metric in stats.index:
                pdf.set_font("Arial", "", 12)
                pdf.cell(
                    0, 10,
                    f"{metric}:",
                    ln=True
                )
                for stat, value in stats.loc[metric].items():
                    pdf.cell(
                        0, 10,
                        f"{stat}: {value:.4f}",
                        ln=True
                    )
                    
        # Save all plots
        self.plot_all_metrics(save=True, show=False)
        self.plot_learning_curves(save=True, show=False)
        self.plot_metric_relationships(save=True, show=False)
        self.plot_metric_embedding(save=True, show=False)
        self.plot_metric_radar(save=True, show=False)
        
        # Add plots to report
        for img in self.output_dir.glob("*.png"):
            pdf.add_page()
            pdf.image(str(img), x=10, y=10, w=190)
            
        pdf.output(output_file)
        
if __name__ == "__main__":
    # Example usage
    visualizer = MetricsVisualizer("logs/cleanup/metrics_latest.csv")
    
    # Generate all visualizations
    visualizer.plot_all_metrics(interactive=True)
    visualizer.plot_learning_curves()
    visualizer.plot_metric_relationships()
    visualizer.plot_metric_embedding()
    visualizer.plot_metric_radar()
    
    # Generate report
    visualizer.generate_report("training_report.pdf")
