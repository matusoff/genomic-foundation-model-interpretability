"""
Activation patching experiments to identify critical model components.
Tests influence of specific positions and features on predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import pandas as pd
from tqdm import tqdm
import logging

try:
    from .model_loader import GenomicModelLoader
except ImportError:
    from .model_loader_simple import SimpleGenomicModelLoader as GenomicModelLoader
from .data_loader import VariantDataLoader

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class ActivationPatcher:
    """Performs activation patching experiments."""
    
    def __init__(
        self,
        model_loader: GenomicModelLoader,
        results_dir: str = "results"
    ):
        """
        Initialize activation patcher.
        
        Args:
            model_loader: Initialized GenomicModelLoader
            results_dir: Directory to save results
        """
        self.model_loader = model_loader
        self.model_loader.load()
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def patch_activations(
        self,
        source_sequence: str,
        target_sequence: str,
        patch_positions: List[int],
        layer: int,
        patch_type: str = "hidden_state"
    ) -> Dict:
        """
        Patch activations from source into target at specified positions.
        
        Args:
            source_sequence: Source sequence (where activations come from)
            target_sequence: Target sequence (where activations are patched)
            patch_positions: List of positions to patch
            layer: Layer to patch
            patch_type: 'hidden_state' or 'attention'
            
        Returns:
            Dictionary with patched outputs and metrics
        """
        # Encode sequences
        source_encoded = self.model_loader.encode_sequence(source_sequence)
        target_encoded = self.model_loader.encode_sequence(target_sequence)
        
        # Get baseline target output
        with torch.no_grad():
            target_output = self.model(**target_encoded, output_hidden_states=True)
            baseline_hidden = target_output.hidden_states[-1]
        
        # Forward pass with patching hook
        patched_hidden = None
        
        def patch_hook(module, input, output):
            """Hook to patch activations."""
            nonlocal patched_hidden
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Clone to avoid in-place modification
            patched = hidden.clone()
            
            # Patch at specified positions
            for pos in patch_positions:
                if pos < hidden.shape[1]:
                    patched[:, pos, :] = source_encoded["input_ids"][:, pos:pos+1]
            
            patched_hidden = patched
            return (patched,) if isinstance(output, tuple) else patched
        
        # Handle different model architectures (ESM vs standard)
        if hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
        elif hasattr(self.model, 'esm') and hasattr(self.model.esm, 'encoder'):
            encoder = self.model.esm.encoder
        else:
            logger.error("Cannot find encoder in model")
            return {"baseline_hidden": None, "patched_hidden": None, "difference": 0.0, "patch_positions": patch_positions}
        
        if hasattr(encoder, 'layer'):
            layer_module = encoder.layer[layer]
        elif hasattr(encoder, 'layers'):
            layer_module = encoder.layers[layer]
        else:
            logger.error(f"Cannot find layer {layer}")
            return {"baseline_hidden": None, "patched_hidden": None, "difference": 0.0, "patch_positions": patch_positions}
        
        handle = layer_module.register_forward_hook(patch_hook)
        
        try:
            with torch.no_grad():
                patched_output = self.model(**target_encoded, output_hidden_states=True)
        finally:
            handle.remove()
        
        # Compute difference
        if patched_hidden is not None:
            diff = (patched_hidden - baseline_hidden).abs().mean().item()
        else:
            diff = 0.0
        
        return {
            "baseline_hidden": baseline_hidden.cpu(),
            "patched_hidden": patched_hidden.cpu() if patched_hidden is not None else None,
            "difference": diff,
            "patch_positions": patch_positions
        }
    
    def ablate_positions(
        self,
        sequence: str,
        ablate_positions: List[int],
        layer: Optional[int] = None
    ) -> Dict:
        """
        Ablate (zero out) specific positions and measure impact.
        
        Args:
            sequence: Input sequence
            ablate_positions: Positions to ablate (in sequence space, not token space)
            layer: Layer to ablate (None for input)
            
        Returns:
            Dictionary with ablation results
        """
        encoded = self.model_loader.encode_sequence(sequence)
        
        # Baseline
        with torch.no_grad():
            baseline_output = self.model(**encoded, output_hidden_states=True)
            # Handle different output structures
            if hasattr(baseline_output, 'hidden_states') and baseline_output.hidden_states:
                if isinstance(baseline_output.hidden_states, tuple):
                    baseline_hidden = baseline_output.hidden_states[-1]
                else:
                    baseline_hidden = baseline_output.hidden_states[-1] if hasattr(baseline_output.hidden_states, '__getitem__') else baseline_output.hidden_states
            elif hasattr(baseline_output, 'last_hidden_state'):
                baseline_hidden = baseline_output.last_hidden_state
            else:
                baseline_hidden = baseline_output.logits
        
        # Ablate at input level
        # Note: Nucleotide Transformer uses k-mer tokenization (~6 nucleotides per token)
        # We need to map from nucleotide positions to token positions
        ablated_input_ids = encoded["input_ids"].clone()
        ablated_attention_mask = encoded["attention_mask"].clone()
        
        # Find which tokens have attention (not padding)
        active_tokens = torch.where(encoded["attention_mask"][0] == 1)[0]
        
        # Approximate mapping: nucleotide position -> token position
        # Nucleotide Transformer uses k-mer tokenization (~6 nt per token)
        # Token 0 is <cls>, actual sequence starts at token 1
        k_mer_size = 6  # Typical for Nucleotide Transformer
        
        for pos in ablate_positions:
            # Map nucleotide position to approximate token position
            # token_pos = (nucleotide_pos / k_mer_size) + 1 (for <cls> offset)
            token_pos = int(pos / k_mer_size) + 1
            
            # Make sure token is in active range
            if token_pos < len(active_tokens):
                actual_token_pos = active_tokens[token_pos].item()
                
                # Use mask token for ablation
                if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
                    ablated_input_ids[0, actual_token_pos] = self.tokenizer.mask_token_id
                else:
                    ablated_input_ids[0, actual_token_pos] = self.tokenizer.pad_token_id
                    ablated_attention_mask[0, actual_token_pos] = 0
        
        ablated_encoded = {**encoded, "input_ids": ablated_input_ids, "attention_mask": ablated_attention_mask}
        
        with torch.no_grad():
            ablated_output = self.model(**ablated_encoded, output_hidden_states=True)
            # Handle different output structures
            if hasattr(ablated_output, 'hidden_states') and ablated_output.hidden_states:
                if isinstance(ablated_output.hidden_states, tuple):
                    ablated_hidden = ablated_output.hidden_states[-1]
                else:
                    ablated_hidden = ablated_output.hidden_states[-1] if hasattr(ablated_output.hidden_states, '__getitem__') else ablated_output.hidden_states
            elif hasattr(ablated_output, 'last_hidden_state'):
                ablated_hidden = ablated_output.last_hidden_state
            else:
                ablated_hidden = ablated_output.logits
        
        # Compute impact
        impact = (baseline_hidden - ablated_hidden).abs().mean().item()
        
        return {
            "baseline_hidden": baseline_hidden.cpu(),
            "ablated_hidden": ablated_hidden.cpu(),
            "impact": impact,
            "ablate_positions": ablate_positions
        }
    
    def systematic_ablation(
        self,
        ref_sequence: str,
        alt_sequence: str,
        variant_pos: int,
        window_size: int = 50,
        step_size: int = 10
    ) -> pd.DataFrame:
        """
        Systematically ablate positions around variant and measure impact.
        
        Args:
            ref_sequence: Reference sequence
            alt_sequence: Alternate sequence
            variant_pos: Variant position
            window_size: Size of ablation window
            step_size: Step size for sliding window
            
        Returns:
            DataFrame with ablation results
        """
        logger.info("Performing systematic ablation analysis")
        
        seq_len = len(ref_sequence)
        results = []
        
        # Test different positions
        positions_to_test = range(
            max(0, variant_pos - window_size),
            min(seq_len, variant_pos + window_size),
            step_size
        )
        
        for pos in tqdm(positions_to_test, desc="Ablating positions"):
            ablate_positions = list(range(pos, min(seq_len, pos + step_size)))
            
            # Ablate in reference
            ref_result = self.ablate_positions(ref_sequence, ablate_positions)
            
            # Ablate in alternate
            alt_result = self.ablate_positions(alt_sequence, ablate_positions)
            
            results.append({
                "position": pos,
                "distance_from_variant": abs(pos - variant_pos),
                "ref_impact": ref_result["impact"],
                "alt_impact": alt_result["impact"],
                "impact_difference": alt_result["impact"] - ref_result["impact"]
            })
        
        df = pd.DataFrame(results)
        return df
    
    def visualize_ablation_results(
        self,
        ablation_df: pd.DataFrame,
        variant_pos: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize systematic ablation results.
        
        Args:
            ablation_df: DataFrame from systematic_ablation
            variant_pos: Variant position
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        positions = ablation_df["position"].values
        ref_impact = ablation_df["ref_impact"].values
        alt_impact = ablation_df["alt_impact"].values
        impact_diff = ablation_df["impact_difference"].values
        
        # Check for valid data
        # Remove NaN values first, then check if any non-zero values exist
        ref_valid = ref_impact[~np.isnan(ref_impact)]
        alt_valid = alt_impact[~np.isnan(alt_impact)]
        has_ref = len(ref_valid) > 0 and not np.allclose(ref_valid, 0, atol=1e-10)
        has_alt = len(alt_valid) > 0 and not np.allclose(alt_valid, 0, atol=1e-10)
        
        # Plot 1: Impact comparison
        if has_ref:
            axes[0].plot(positions, ref_impact, 'b-o', label='Reference', alpha=0.7, linewidth=2, markersize=4)
        if has_alt:
            axes[0].plot(positions, alt_impact, 'r-o', label='Alternate', alpha=0.7, linewidth=2, markersize=4)
        axes[0].axvline(x=variant_pos, color='green', linestyle='--', linewidth=2, label='Variant Position')
        axes[0].set_xlabel('Ablated Position', fontsize=12)
        axes[0].set_ylabel('Impact (Mean Absolute Difference)', fontsize=12)
        axes[0].set_title('Impact of Position Ablation', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Impact difference
        axes[1].bar(positions, impact_diff, color='purple', alpha=0.7, width=5)
        axes[1].axvline(x=variant_pos, color='green', linestyle='--', linewidth=2, label='Variant Position')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_xlabel('Ablated Position', fontsize=12)
        axes[1].set_ylabel('Impact Difference (Alt - Ref)', fontsize=12)
        axes[1].set_title('Differential Impact of Ablation', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ablation visualization to {save_path}")
            plt.close(fig)  # Close figure to free memory and prevent accumulation
        
        return fig
    
    def analyze_critical_positions(
        self,
        ref_sequence: str,
        alt_sequence: str,
        variant_pos: int,
        top_k: int = 10
    ) -> Dict:
        """
        Identify most critical positions for variant prediction.
        
        Args:
            ref_sequence: Reference sequence
            alt_sequence: Alternate sequence
            variant_pos: Variant position
            top_k: Number of top positions to return
            
        Returns:
            Dictionary with critical positions and analysis
        """
        logger.info("Identifying critical positions")
        
        # Perform systematic ablation
        ablation_df = self.systematic_ablation(
            ref_sequence,
            alt_sequence,
            variant_pos,
            window_size=100,
            step_size=5
        )
        
        # Find positions with highest impact difference
        top_positions = ablation_df.nlargest(top_k, "impact_difference")
        
        # Check for valid data
        if len(ablation_df) == 0 or ablation_df["ref_impact"].isna().all() or ablation_df["alt_impact"].isna().all():
            logger.warning("No valid ablation data to visualize")
            return {
                "top_critical_positions": pd.DataFrame(),
                "ablation_df": ablation_df
            }
        
        # Visualize (use timestamp to avoid overwriting)
        import time
        timestamp = int(time.time() * 1000)  # milliseconds
        
        # Create fresh figure for timestamped version
        fig1 = self.visualize_ablation_results(
            ablation_df,
            variant_pos,
            save_path=str(self.figures_dir / f"ablation_analysis_{timestamp}.png")
        )
        
        # Create fresh figure for "latest" version (don't reuse fig1)
        fig2 = self.visualize_ablation_results(
            ablation_df,
            variant_pos,
            save_path=str(self.figures_dir / "ablation_analysis.png")
        )
        
        return {
            "ablation_results": ablation_df,
            "top_critical_positions": top_positions,
            "variant_position": variant_pos
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test activation patching
    model_loader = GenomicModelLoader()
    patcher = ActivationPatcher(model_loader)
    
    # Test sequences
    ref_seq = "ATGC" * 50
    alt_seq = "ATGC" * 49 + "ATCC"
    variant_pos = 196
    
    # Test ablation
    result = patcher.ablate_positions(ref_seq, [variant_pos])
    print(f"Ablation impact: {result['impact']}")
    
    # Systematic ablation
    ablation_df = patcher.systematic_ablation(ref_seq, alt_seq, variant_pos)
    print(f"\nAblation results shape: {ablation_df.shape}")
    print(ablation_df.head())

