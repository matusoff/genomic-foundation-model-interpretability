"""
Attention visualization and analysis for genomic foundation models.
Analyzes attention patterns to understand how the model processes variants.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

try:
    from .model_loader import GenomicModelLoader
except ImportError:
    from .model_loader_simple import SimpleGenomicModelLoader as GenomicModelLoader
from .data_loader import VariantDataLoader
import logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class AttentionAnalyzer:
    """Analyzer for attention patterns in genomic models."""
    
    def __init__(
        self,
        model_loader: GenomicModelLoader,
        results_dir: str = "results"
    ):
        """
        Initialize attention analyzer.
        
        Args:
            model_loader: Initialized GenomicModelLoader
            results_dir: Directory to save results
        """
        self.model_loader = model_loader
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_attentions(
        self,
        sequence: str,
        max_length: int = 512
    ) -> Dict:
        """
        Extract attention weights for a sequence.
        
        Args:
            sequence: DNA sequence
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with attention weights and metadata
        """
        outputs = self.model_loader.forward_with_attentions(sequence, max_length)
        
        attentions = outputs["attentions"]
        
        if attentions is None:
            logger.warning("No attentions available in model outputs")
            # Create dummy attentions for demo
            num_layers = 6  # Default for ESM
            num_heads = 8
            seq_len = len(sequence)
            attentions = torch.zeros(num_layers, 1, num_heads, seq_len, seq_len)
        else:
            # Handle tuple or tensor
            if isinstance(attentions, tuple):
                attentions = torch.stack(attentions)
            elif not isinstance(attentions, torch.Tensor):
                attentions = torch.stack(list(attentions))
        
        # Remove batch dimension if present
        if attentions.dim() == 5:
            attentions = attentions.squeeze(1)  # [num_layers, num_heads, seq_len, seq_len]
        
        return {
            "attentions": attentions.cpu().numpy(),
            "sequence": sequence,
            "input_ids": outputs["input_ids"].cpu().numpy()
        }
    
    def compute_attention_statistics(
        self,
        attentions: np.ndarray
    ) -> Dict:
        """
        Compute statistics over attention weights.
        
        Args:
            attentions: Attention array [num_layers, num_heads, seq_len, seq_len]
            
        Returns:
            Dictionary with statistics
        """
        num_layers, num_heads, seq_len, _ = attentions.shape
        
        stats = {
            "mean_attention": np.mean(attentions, axis=(2, 3)),  # [layers, heads]
            "max_attention": np.max(attentions, axis=(2, 3)),
            "entropy": self._compute_attention_entropy(attentions),
            "attention_to_variant": None  # Will be set if variant position known
        }
        
        return stats
    
    def _compute_attention_entropy(
        self,
        attentions: np.ndarray
    ) -> np.ndarray:
        """Compute entropy of attention distributions."""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        attentions_safe = attentions + eps
        
        # Normalize to get probability distributions
        attentions_norm = attentions_safe / (attentions_safe.sum(axis=-1, keepdims=True) + eps)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -np.sum(attentions_norm * np.log(attentions_norm + eps), axis=-1)
        
        # Average over sequence positions
        entropy_mean = np.mean(entropy, axis=-1)  # [layers, heads]
        
        return entropy_mean
    
    def visualize_attention_heatmap(
        self,
        attentions: np.ndarray,
        layer: int,
        head: int,
        sequence: str,
        variant_pos: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention heatmap for a specific layer and head.
        
        Args:
            attentions: Attention array
            layer: Layer index
            head: Head index
            sequence: DNA sequence
            variant_pos: Position of variant in sequence (optional)
            save_path: Path to save figure
        """
        attn_matrix = attentions[layer, head]
        
        # Truncate to actual sequence length (remove padding)
        seq_len = len(sequence)
        attn_matrix = attn_matrix[:seq_len, :seq_len]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Highlight variant position if provided
        if variant_pos is not None:
            ax.axhline(y=variant_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvline(x=variant_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Query Position', fontsize=12)
        ax.set_ylabel('Key Position', fontsize=12)
        ax.set_title(f'Attention Heatmap - Layer {layer}, Head {head}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {save_path}")
        
        return fig
    
    def compare_variant_attentions(
        self,
        ref_sequence: str,
        alt_sequence: str,
        variant_pos: int,
        layers: Optional[List[int]] = None,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Compare attention patterns between reference and alternate sequences.
        
        Args:
            ref_sequence: Reference sequence
            alt_sequence: Alternate sequence
            variant_pos: Position of variant
            layers: Specific layers to analyze (all if None)
            save_dir: Directory to save figures
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing attention patterns between ref and alt sequences")
        
        # Initialize variables
        ref_attn_array = None
        alt_attn_array = None
        num_layers = 6
        num_heads = 8
        
        try:
            # Extract attentions
            ref_attn = self.extract_attentions(ref_sequence)
            alt_attn = self.extract_attentions(alt_sequence)
            
            ref_attn_array = ref_attn.get("attentions") if isinstance(ref_attn, dict) else None
            alt_attn_array = alt_attn.get("attentions") if isinstance(alt_attn, dict) else None
            
            if ref_attn_array is None or alt_attn_array is None or (hasattr(ref_attn_array, 'size') and ref_attn_array.size == 0) or (isinstance(ref_attn_array, np.ndarray) and ref_attn_array.size == 0):
                logger.warning("No attentions available, using hidden states as proxy")
                # Fallback: use hidden states to compute attention-like patterns
                try:
                    ref_outputs = self.model_loader.forward_with_attentions(ref_sequence)
                    alt_outputs = self.model_loader.forward_with_attentions(alt_sequence)
                    
                    # Get hidden states - handle ESM model structure
                    ref_hidden = None
                    if "last_hidden_state" in ref_outputs and ref_outputs["last_hidden_state"] is not None:
                        ref_hidden = ref_outputs["last_hidden_state"]
                    elif "hidden_states" in ref_outputs and ref_outputs["hidden_states"] is not None:
                        hidden_states = ref_outputs["hidden_states"]
                        if isinstance(hidden_states, tuple) and len(hidden_states) > 0:
                            ref_hidden = hidden_states[-1]
                        elif hasattr(hidden_states, '__getitem__') and len(hidden_states) > 0:
                            ref_hidden = hidden_states[-1]
                    
                    alt_hidden = None
                    if "last_hidden_state" in alt_outputs and alt_outputs["last_hidden_state"] is not None:
                        alt_hidden = alt_outputs["last_hidden_state"]
                    elif "hidden_states" in alt_outputs and alt_outputs["hidden_states"] is not None:
                        hidden_states = alt_outputs["hidden_states"]
                        if isinstance(hidden_states, tuple) and len(hidden_states) > 0:
                            alt_hidden = hidden_states[-1]
                        elif hasattr(hidden_states, '__getitem__') and len(hidden_states) > 0:
                            alt_hidden = hidden_states[-1]
                    
                    if ref_hidden is not None and alt_hidden is not None:
                        # Compute similarity matrix as proxy for attention
                        ref_attn_array = self._hidden_to_attention_proxy(ref_hidden)
                        alt_attn_array = self._hidden_to_attention_proxy(alt_hidden)
                    else:
                        raise ValueError(f"Could not extract hidden states. ref_hidden: {ref_hidden is not None}, alt_hidden: {alt_hidden is not None}")
                except Exception as e2:
                    logger.warning(f"Could not use hidden states as proxy: {e2}")
                    # Create dummy arrays
                    num_layers = 6
                    num_heads = 8
                    seq_len = min(len(ref_sequence), 100)
                    ref_attn_array = np.zeros((num_layers, num_heads, seq_len, seq_len))
                    alt_attn_array = np.zeros((num_layers, num_heads, seq_len, seq_len))
            
            num_layers = ref_attn_array.shape[0]
            if layers is None:
                layers = list(range(num_layers))
            
            # Compute differences
            attn_diff = alt_attn_array - ref_attn_array
            
            # Analyze attention to variant position
            # Convert nucleotide position to token position (k-mer tokenization: ~6 nt per token)
            # Token 0 is <cls>, actual sequence starts at token 1
            k_mer_size = 6
            token_variant_pos = min(variant_pos // k_mer_size + 1, ref_attn_array.shape[-1] - 1)
            
            logger.info(f"Variant nucleotide pos: {variant_pos}, token pos: {token_variant_pos}")
            
            if token_variant_pos < ref_attn_array.shape[-1]:
                variant_attention_ref = ref_attn_array[:, :, :, token_variant_pos].mean(axis=2)  # [layers, heads]
                variant_attention_alt = alt_attn_array[:, :, :, token_variant_pos].mean(axis=2)
                variant_attention_diff = variant_attention_alt - variant_attention_ref
            else:
                # Fallback to center if still out of range
                center_pos = ref_attn_array.shape[-1] // 2
                logger.warning(f"Token position {token_variant_pos} out of range, using center {center_pos}")
                variant_attention_ref = ref_attn_array[:, :, :, center_pos].mean(axis=2)
                variant_attention_alt = alt_attn_array[:, :, :, center_pos].mean(axis=2)
                variant_attention_diff = variant_attention_alt - variant_attention_ref
        except Exception as e:
            logger.error(f"Error in attention comparison: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return dummy data structure
            variant_attention_ref = np.zeros((num_layers, num_heads))
            variant_attention_alt = np.zeros((num_layers, num_heads))
            variant_attention_diff = np.zeros((num_layers, num_heads))
            attn_diff = np.zeros((num_layers, num_heads, 100, 100))
            if ref_attn_array is None:
                ref_attn_array = np.zeros((num_layers, num_heads, 100, 100))
            if alt_attn_array is None:
                alt_attn_array = np.zeros((num_layers, num_heads, 100, 100))
        
        results = {
            "attention_diff": attn_diff,
            "variant_attention_ref": variant_attention_ref,
            "variant_attention_alt": variant_attention_alt,
            "variant_attention_diff": variant_attention_diff,
            "ref_attentions": ref_attn_array,
            "alt_attentions": alt_attn_array
        }
        
        # Visualize differences
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Plot variant attention differences across layers
            self._plot_variant_attention_comparison(
                variant_attention_ref,
                variant_attention_alt,
                save_path / "variant_attention_comparison.png"
            )
            
            # Plot attention difference heatmaps for key layers
            for layer in layers[:3]:  # First 3 layers
                for head in [0, 4]:  # Sample heads
                    self._plot_attention_difference(
                        attn_diff[layer, head],
                        layer,
                        head,
                        variant_pos,
                        save_path / f"attention_diff_layer{layer}_head{head}.png"
                    )
        
        return results
    
    def _plot_variant_attention_comparison(
        self,
        ref_attn: np.ndarray,
        alt_attn: np.ndarray,
        save_path: Path
    ):
        """Plot comparison of attention to variant position."""
        if ref_attn is None or alt_attn is None:
            logger.warning("Cannot plot: missing attention data")
            return
        
        if ref_attn.ndim == 1:
            # Single layer case
            num_layers = 1
            ref_attn = ref_attn.reshape(1, -1)
            alt_attn = alt_attn.reshape(1, -1)
        else:
            num_layers, num_heads = ref_attn.shape
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Average across heads
        ref_mean = ref_attn.mean(axis=1) if ref_attn.ndim > 1 else ref_attn
        alt_mean = alt_attn.mean(axis=1) if alt_attn.ndim > 1 else alt_attn
        
        layers = np.arange(len(ref_mean))
        
        axes[0].plot(layers, ref_mean, 'b-o', label='Reference', linewidth=2)
        axes[0].plot(layers, alt_mean, 'r-o', label='Alternate', linewidth=2)
        axes[0].set_xlabel('Layer', fontsize=12)
        axes[0].set_ylabel('Mean Attention to Variant', fontsize=12)
        axes[0].set_title('Attention to Variant Position', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Difference
        diff = alt_mean - ref_mean
        axes[1].bar(layers, diff, color='purple', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_xlabel('Layer', fontsize=12)
        axes[1].set_ylabel('Attention Difference', fontsize=12)
        axes[1].set_title('Difference in Attention (Alt - Ref)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_difference(
        self,
        attn_diff: np.ndarray,
        layer: int,
        head: int,
        variant_pos: int,
        save_path: Path
    ):
        """Plot attention difference heatmap."""
        seq_len = attn_diff.shape[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use TwoSlopeNorm for centered colormap around 0
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
        im = ax.imshow(attn_diff, cmap='RdBu_r', norm=norm, aspect='auto')
        
        # Highlight variant position
        ax.axhline(y=variant_pos, color='yellow', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(x=variant_pos, color='yellow', linestyle='--', linewidth=2, alpha=0.8)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Difference (Alt - Ref)', rotation=270, labelpad=20)
        
        ax.set_xlabel('Query Position', fontsize=12)
        ax.set_ylabel('Key Position', fontsize=12)
        ax.set_title(f'Attention Difference - Layer {layer}, Head {head}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _hidden_to_attention_proxy(self, hidden_states) -> np.ndarray:
        """Convert hidden states to attention-like similarity matrix."""
        import torch
        
        if isinstance(hidden_states, torch.Tensor):
            hidden = hidden_states.cpu().numpy()
        else:
            hidden = np.array(hidden_states)
        
        # Remove batch dimension if present
        if hidden.ndim == 3:
            hidden = hidden[0]  # [seq_len, hidden_dim]
        elif hidden.ndim == 2:
            pass  # Already [seq_len, hidden_dim]
        else:
            raise ValueError(f"Unexpected hidden states shape: {hidden.shape}")
        
        # Limit sequence length for efficiency
        max_seq_len = min(512, hidden.shape[0])
        hidden = hidden[:max_seq_len]
        
        # Compute similarity matrix (cosine similarity) - vectorized for efficiency
        seq_len = hidden.shape[0]
        norms = np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-10
        normalized = hidden / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Reshape to match attention format [layers, heads, seq, seq]
        similarity = similarity_matrix[np.newaxis, np.newaxis, :, :]
        
        return similarity


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test attention analysis
    model_loader = GenomicModelLoader()
    analyzer = AttentionAnalyzer(model_loader)
    
    # Test sequence
    test_seq = "ATGCATGCATGC" * 20
    attn_data = analyzer.extract_attentions(test_seq)
    
    print(f"Attention shape: {attn_data['attentions'].shape}")
    
    # Visualize
    analyzer.visualize_attention_heatmap(
        attn_data["attentions"],
        layer=0,
        head=0,
        sequence=test_seq,
        save_path="test_attention.png"
    )

