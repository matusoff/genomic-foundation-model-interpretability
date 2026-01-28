"""
Circuit analysis and ablation studies.
Identifies functional circuits (groups of neurons/heads) that work together.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import logging
from itertools import product

try:
    from .model_loader import GenomicModelLoader
except ImportError:
    from .model_loader_simple import SimpleGenomicModelLoader as GenomicModelLoader
from .data_loader import VariantDataLoader

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class CircuitAnalyzer:
    """Analyzer for identifying functional circuits in the model."""
    
    def __init__(
        self,
        model_loader: GenomicModelLoader,
        results_dir: str = "results"
    ):
        """
        Initialize circuit analyzer.
        
        Args:
            model_loader: Initialized GenomicModelLoader
            results_dir: Directory to save results
        """
        self.model_loader = model_loader
        self.model_loader.load()
        self.model = model_loader.model
        self.device = model_loader.device
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model info
        info = model_loader.get_model_info()
        self.num_layers = info["num_layers"]
        self.num_heads = info["num_heads"]
    
    def ablate_attention_head(
        self,
        sequence: str,
        layer: int,
        head: int
    ) -> Dict:
        """
        Ablate a specific attention head by zeroing its output.
        
        Args:
            sequence: Input sequence
            layer: Layer index
            head: Head index
            
        Returns:
            Dictionary with ablation results
        """
        encoded = self.model_loader.encode_sequence(sequence)
        
        # Baseline
        with torch.no_grad():
            baseline_output = self.model(**encoded, output_hidden_states=True)
            baseline_hidden = baseline_output.hidden_states[-1]
        
        # Hook to ablate attention head
        ablated_output = None
        
        def ablation_hook(module, input, output):
            """Hook to zero out specific attention head."""
            nonlocal ablated_output
            # output is tuple: (hidden_states, attention_weights) or just hidden_states
            if isinstance(output, tuple):
                hidden, attn = output[0], output[1] if len(output) > 1 else None
            else:
                hidden = output
                attn = None
            
            # Clone to avoid in-place modification
            ablated_hidden = hidden.clone()
            
            # The attention output has shape [batch, seq_len, hidden_dim]
            # hidden_dim = num_heads * head_dim
            # We need to zero out the specific head's contribution
            batch_size, seq_len, hidden_dim = ablated_hidden.shape
            head_dim = hidden_dim // self.num_heads
            
            # Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
            ablated_hidden = ablated_hidden.view(batch_size, seq_len, self.num_heads, head_dim)
            
            # Zero out the specific head
            ablated_hidden[:, :, head, :] = 0.0
            
            # Reshape back to [batch, seq_len, hidden_dim]
            ablated_hidden = ablated_hidden.view(batch_size, seq_len, hidden_dim)
            
            ablated_output = ablated_hidden
            return (ablated_output, attn) if attn is not None else ablated_output
        
        # Handle different model architectures (ESM vs standard)
        if hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
        elif hasattr(self.model, 'esm') and hasattr(self.model.esm, 'encoder'):
            encoder = self.model.esm.encoder
        else:
            logger.error("Cannot find encoder in model")
            return {"baseline_hidden": None, "ablated_hidden": None, "impact": 0.0}
        
        # Get attention layer
        if hasattr(encoder, 'layer'):
            attention_layer = encoder.layer[layer].attention
        elif hasattr(encoder, 'layers'):
            attention_layer = encoder.layers[layer].attention
        else:
            logger.error(f"Cannot find layer {layer} in encoder")
            return {"baseline_hidden": None, "ablated_hidden": None, "impact": 0.0}
        
        handle = attention_layer.register_forward_hook(ablation_hook)
        
        try:
            with torch.no_grad():
                ablated_output_full = self.model(**encoded, output_hidden_states=True)
                # Get the final hidden states after ablation
                ablated_final_hidden = ablated_output_full.hidden_states[-1]
        finally:
            handle.remove()
        
        # Compute impact by comparing final hidden states
        if ablated_output is not None and ablated_final_hidden is not None:
            impact = (baseline_hidden - ablated_final_hidden).abs().mean().item()
        else:
            impact = 0.0
        
        return {
            "baseline_hidden": baseline_hidden.cpu(),
            "ablated_hidden": ablated_final_hidden.cpu() if ablated_final_hidden is not None else None,
            "impact": impact
        }
    
    def ablate_layer(
        self,
        sequence: str,
        layer: int
    ) -> Dict:
        """
        Ablate an entire layer by zeroing its output.
        
        Args:
            sequence: Input sequence
            layer: Layer index
            
        Returns:
            Dictionary with ablation results
        """
        encoded = self.model_loader.encode_sequence(sequence)
        
        # Baseline
        with torch.no_grad():
            baseline_output = self.model(**encoded, output_hidden_states=True)
            baseline_hidden = baseline_output.hidden_states[-1]
        
        # Hook to zero layer output
        def layer_ablation_hook(module, input, output):
            """Hook to zero layer output."""
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        
        # Handle different model architectures
        if hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
        elif hasattr(self.model, 'esm') and hasattr(self.model.esm, 'encoder'):
            encoder = self.model.esm.encoder
        else:
            logger.error("Cannot find encoder in model")
            return {"baseline_hidden": None, "ablated_hidden": None, "impact": 0.0}
        
        if hasattr(encoder, 'layer'):
            layer_module = encoder.layer[layer]
        elif hasattr(encoder, 'layers'):
            layer_module = encoder.layers[layer]
        else:
            logger.error(f"Cannot find layer {layer}")
            return {"baseline_hidden": None, "ablated_hidden": None, "impact": 0.0}
        
        handle = layer_module.register_forward_hook(layer_ablation_hook)
        
        try:
            with torch.no_grad():
                ablated_output = self.model(**encoded, output_hidden_states=True)
                ablated_hidden = ablated_output.hidden_states[-1]
        finally:
            handle.remove()
        
        impact = (baseline_hidden - ablated_hidden).abs().mean().item()
        
        return {
            "baseline_hidden": baseline_hidden.cpu(),
            "ablated_hidden": ablated_hidden.cpu(),
            "impact": impact
        }
    
    def systematic_head_ablation(
        self,
        ref_sequence: str,
        alt_sequence: str,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Systematically ablate attention heads and measure impact.
        
        Args:
            ref_sequence: Reference sequence
            alt_sequence: Alternate sequence
            layers: Layers to test (all if None)
            heads: Heads to test (all if None)
            
        Returns:
            DataFrame with ablation results
        """
        logger.info("Performing systematic head ablation")
        
        if layers is None:
            layers = list(range(self.num_layers))
        if heads is None:
            heads = list(range(self.num_heads))
        
        results = []
        
        for layer, head in tqdm(
            product(layers, heads),
            desc="Ablating heads",
            total=len(layers) * len(heads)
        ):
            # Ablate in reference
            ref_result = self.ablate_attention_head(ref_sequence, layer, head)
            
            # Ablate in alternate
            alt_result = self.ablate_attention_head(alt_sequence, layer, head)
            
            results.append({
                "layer": layer,
                "head": head,
                "ref_impact": ref_result["impact"],
                "alt_impact": alt_result["impact"],
                "impact_difference": alt_result["impact"] - ref_result["impact"]
            })
        
        df = pd.DataFrame(results)
        return df
    
    def find_critical_circuits(
        self,
        ref_sequence: str,
        alt_sequence: str,
        top_k: int = 10,
        sample_layers: Optional[int] = None,
        sample_heads: Optional[int] = None
    ) -> Dict:
        """
        Find critical circuits (layer-head combinations) for variant prediction.
        
        Args:
            ref_sequence: Reference sequence
            alt_sequence: Alternate sequence
            top_k: Number of top circuits to return
            sample_layers: Only test this many layers (e.g., 10 = last 10 layers). None = all layers
            sample_heads: Only test this many heads (e.g., 8 = every other head). None = all heads
            
        Returns:
            Dictionary with critical circuits
        """
        logger.info("Finding critical circuits")
        
        # Optionally sample layers/heads to speed up
        layers = None
        heads = None
        
        if sample_layers is not None:
            # Test only the last N layers (typically more important)
            layers = list(range(max(0, self.num_layers - sample_layers), self.num_layers))
            logger.info(f"Sampling last {sample_layers} layers: {layers[0]}-{layers[-1]}")
        
        if sample_heads is not None:
            # Sample every N-th head
            step = max(1, self.num_heads // sample_heads)
            heads = list(range(0, self.num_heads, step))[:sample_heads]
            logger.info(f"Sampling {len(heads)} heads: {heads}")
        
        # Perform systematic ablation
        ablation_df = self.systematic_head_ablation(ref_sequence, alt_sequence, layers=layers, heads=heads)
        
        # Find circuits with highest impact difference
        top_circuits = ablation_df.nlargest(top_k, "impact_difference")
        
        # Check for valid data
        if len(ablation_df) == 0 or ablation_df["impact_difference"].isna().all():
            logger.warning("No valid circuit data to visualize")
            return {
                "top_circuits": pd.DataFrame(),
                "ablation_df": ablation_df
            }
        
        # Visualize (use timestamp to avoid overwriting)
        import time
        timestamp = int(time.time() * 1000)  # milliseconds
        
        # Create fresh figure for timestamped version
        fig1 = self.visualize_circuit_analysis(
            ablation_df,
            top_circuits,
            save_path=str(self.figures_dir / f"circuit_analysis_{timestamp}.png")
        )
        
        # Create fresh figure for "latest" version
        fig2 = self.visualize_circuit_analysis(
            ablation_df,
            top_circuits,
            save_path=str(self.figures_dir / "circuit_analysis.png")
        )
        
        return {
            "ablation_results": ablation_df,
            "top_circuits": top_circuits,
            "critical_layers": top_circuits["layer"].unique().tolist(),
            "critical_heads": top_circuits["head"].unique().tolist()
        }
    
    def visualize_circuit_analysis(
        self,
        ablation_df: pd.DataFrame,
        top_circuits: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Visualize circuit analysis results.
        
        Args:
            ablation_df: Full ablation results
            top_circuits: Top circuits DataFrame
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Check for valid data
        if len(ablation_df) == 0 or ablation_df["impact_difference"].isna().all():
            logger.warning("No valid circuit data to visualize - all impact differences are NaN")
            # Create empty plots with message
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            return fig
        
        # Reshape for heatmap
        impact_diff_matrix = ablation_df.pivot(
            index="layer",
            columns="head",
            values="impact_difference"
        )
        
        # Plot 1: Impact difference heatmap
        sns.heatmap(
            impact_diff_matrix,
            cmap='RdBu_r',
            center=0,
            ax=axes[0, 0],
            cbar_kws={'label': 'Impact Difference'}
        )
        axes[0, 0].set_title('Impact Difference by Layer and Head', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Head', fontsize=12)
        axes[0, 0].set_ylabel('Layer', fontsize=12)
        
        # Plot 2: Top circuits
        top_circuits_sorted = top_circuits.sort_values("impact_difference", ascending=True)
        y_pos = np.arange(len(top_circuits_sorted))
        axes[0, 1].barh(
            y_pos,
            top_circuits_sorted["impact_difference"],
            color='purple',
            alpha=0.7
        )
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(
            [f"L{l}H{h}" for l, h in zip(
                top_circuits_sorted["layer"],
                top_circuits_sorted["head"]
            )]
        )
        axes[0, 1].set_xlabel('Impact Difference', fontsize=12)
        axes[0, 1].set_title('Top Critical Circuits', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Impact by layer
        layer_impact = ablation_df.groupby("layer")["impact_difference"].mean()
        axes[1, 0].bar(layer_impact.index, layer_impact.values, color='steelblue', alpha=0.7)
        axes[1, 0].set_xlabel('Layer', fontsize=12)
        axes[1, 0].set_ylabel('Mean Impact Difference', fontsize=12)
        axes[1, 0].set_title('Impact by Layer', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Impact by head
        head_impact = ablation_df.groupby("head")["impact_difference"].mean()
        axes[1, 1].bar(head_impact.index, head_impact.values, color='coral', alpha=0.7)
        axes[1, 1].set_xlabel('Head', fontsize=12)
        axes[1, 1].set_ylabel('Mean Impact Difference', fontsize=12)
        axes[1, 1].set_title('Impact by Head', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved circuit analysis to {save_path}")
            plt.close(fig)  # Close figure to free memory and prevent accumulation
        
        return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test circuit analysis
    model_loader = GenomicModelLoader()
    analyzer = CircuitAnalyzer(model_loader)
    
    # Test sequences
    ref_seq = "ATGC" * 50
    alt_seq = "ATGC" * 49 + "ATCC"
    
    # Find critical circuits
    circuits = analyzer.find_critical_circuits(ref_seq, alt_seq, top_k=5)
    print(f"\nTop circuits:")
    print(circuits["top_circuits"])

