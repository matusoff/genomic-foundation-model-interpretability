"""
Aggregate interpretability results across all variants.
Creates averaged visualizations for ablation, circuit, and attention analyses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def aggregate_ablation_results(results_df: pd.DataFrame, output_dir: Path):
    """Aggregate ablation results across variants."""
    logger.info("Aggregating ablation results...")
    
    # Collect all ablation data
    all_positions = []
    all_ref_impacts = []
    all_alt_impacts = []
    all_impact_diffs = []
    
    for idx, row in results_df.iterrows():
        if row.get("critical_positions") is None:
            continue
        
        # Parse critical_positions (stored as list of dicts)
        try:
            import ast
            if isinstance(row["critical_positions"], str):
                positions_data = ast.literal_eval(row["critical_positions"])
            else:
                positions_data = row["critical_positions"]
            
            if not isinstance(positions_data, list):
                continue
                
            for pos_data in positions_data:
                if isinstance(pos_data, dict):
                    all_positions.append(pos_data.get("position", 0))
                    all_ref_impacts.append(pos_data.get("ref_impact", 0))
                    all_alt_impacts.append(pos_data.get("alt_impact", 0))
                    all_impact_diffs.append(pos_data.get("impact_difference", 0))
        except Exception as e:
            logger.warning(f"Error parsing ablation data for variant {idx}: {e}")
            continue
    
    if len(all_positions) == 0:
        logger.warning("No ablation data to aggregate")
        return
    
    # Create DataFrame
    ablation_df = pd.DataFrame({
        "position": all_positions,
        "ref_impact": all_ref_impacts,
        "alt_impact": all_alt_impacts,
        "impact_difference": all_impact_diffs
    })
    
    # Group by position and compute mean
    aggregated = ablation_df.groupby("position").agg({
        "ref_impact": "mean",
        "alt_impact": "mean",
        "impact_difference": "mean"
    }).reset_index()
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    positions = aggregated["position"].values
    ref_impact = aggregated["ref_impact"].values
    alt_impact = aggregated["alt_impact"].values
    impact_diff = aggregated["impact_difference"].values
    
    # Plot 1: Impact comparison
    axes[0].plot(positions, ref_impact, 'b-o', label='Reference (Mean)', alpha=0.7, linewidth=2, markersize=4)
    axes[0].plot(positions, alt_impact, 'r-o', label='Alternate (Mean)', alpha=0.7, linewidth=2, markersize=4)
    axes[0].set_xlabel('Ablated Position', fontsize=12)
    axes[0].set_ylabel('Impact (Mean Absolute Difference)', fontsize=12)
    axes[0].set_title('Average Impact of Position Ablation Across Variants', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Impact difference
    axes[1].bar(positions, impact_diff, color='purple', alpha=0.7, width=5)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Ablated Position', fontsize=12)
    axes[1].set_ylabel('Impact Difference (Alt - Ref)', fontsize=12)
    axes[1].set_title('Average Differential Impact of Ablation', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / "ablation_analysis_aggregated.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved aggregated ablation analysis to {save_path}")


def aggregate_attention_results(results_df: pd.DataFrame, output_dir: Path, recompute: bool = False):
    """Aggregate attention analysis results across variants.
    
    Note: CSV only stores summary stats. This creates a summary from available data.
    For full layer-by-layer aggregation, set recompute=True (slower).
    """
    logger.info("Aggregating attention results...")
    
    if recompute:
        # Recompute full attention arrays for aggregation
        logger.info("Recomputing attention for full aggregation (this may take a while)...")
        # This would require loading model and data - skip for now
        logger.warning("Full recomputation not implemented. Using summary statistics instead.")
    
    # Collect summary statistics from CSV
    mean_diffs = []
    max_diffs = []
    
    for idx, row in results_df.iterrows():
        if row.get("attention") is None:
            continue
        
        try:
            import ast
            if isinstance(row["attention"], str):
                attn_data = ast.literal_eval(row["attention"])
            else:
                attn_data = row["attention"]
            
            if isinstance(attn_data, dict):
                if "variant_attention_diff" in attn_data:
                    mean_diffs.append(attn_data["variant_attention_diff"])
                if "max_attention_diff" in attn_data:
                    max_diffs.append(attn_data["max_attention_diff"])
        except Exception as e:
            continue
    
    if len(mean_diffs) == 0:
        logger.warning("No attention data available for aggregation")
        logger.info("Individual variant attention plots are available in variant_*_attention/ folders")
        return
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Distribution of mean attention differences
    axes[0].hist(mean_diffs, bins=min(20, len(mean_diffs)), color='steelblue', alpha=0.7, edgecolor='black')
    mean_val = np.mean(mean_diffs)
    axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    axes[0].set_xlabel('Mean Attention Difference (Alt - Ref)', fontsize=12)
    axes[0].set_ylabel('Number of Variants', fontsize=12)
    axes[0].set_title('Distribution of Attention Differences Across Variants', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Max attention differences
    if len(max_diffs) > 0:
        axes[1].hist(max_diffs, bins=min(20, len(max_diffs)), color='coral', alpha=0.7, edgecolor='black')
        max_mean = np.mean(max_diffs)
        axes[1].axvline(max_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {max_mean:.4f}')
        axes[1].set_xlabel('Max Attention Difference (Alt - Ref)', fontsize=12)
        axes[1].set_ylabel('Number of Variants', fontsize=12)
        axes[1].set_title('Distribution of Max Attention Differences', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    save_path = output_dir / "attention_analysis_aggregated.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved aggregated attention summary to {save_path}")
    logger.info(f"  Mean attention diff: {mean_val:.4f} ± {np.std(mean_diffs):.4f}")
    if len(max_diffs) > 0:
        logger.info(f"  Max attention diff: {max_mean:.4f} ± {np.std(max_diffs):.4f}")
    logger.info("  Note: For layer-by-layer aggregation, use individual variant plots")


def aggregate_circuit_results(results_df: pd.DataFrame, output_dir: Path):
    """Aggregate circuit analysis results across variants."""
    logger.info("Aggregating circuit results...")
    
    # Collect all circuit data
    all_layers = []
    all_heads = []
    all_ref_impacts = []
    all_alt_impacts = []
    all_impact_diffs = []
    
    for idx, row in results_df.iterrows():
        if row.get("circuits") is None:
            continue
        
        try:
            import ast
            if isinstance(row["circuits"], str):
                circuits_data = ast.literal_eval(row["circuits"])
            else:
                circuits_data = row["circuits"]
            
            if not isinstance(circuits_data, list):
                continue
                
            for circuit in circuits_data:
                if isinstance(circuit, dict):
                    all_layers.append(circuit.get("layer", 0))
                    all_heads.append(circuit.get("head", 0))
                    all_ref_impacts.append(circuit.get("ref_impact", 0))
                    all_alt_impacts.append(circuit.get("alt_impact", 0))
                    all_impact_diffs.append(circuit.get("impact_difference", 0))
        except Exception as e:
            logger.warning(f"Error parsing circuit data for variant {idx}: {e}")
            continue
    
    if len(all_layers) == 0:
        logger.warning("No circuit data to aggregate")
        return
    
    # Create DataFrame
    circuit_df = pd.DataFrame({
        "layer": all_layers,
        "head": all_heads,
        "ref_impact": all_ref_impacts,
        "alt_impact": all_alt_impacts,
        "impact_difference": all_impact_diffs
    })
    
    # Group by layer-head and compute mean
    aggregated = circuit_df.groupby(["layer", "head"]).agg({
        "ref_impact": "mean",
        "alt_impact": "mean",
        "impact_difference": "mean"
    }).reset_index()
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Reshape for heatmap
    impact_diff_matrix = aggregated.pivot(
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
    axes[0, 0].set_title('Average Impact Difference by Layer and Head', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Head', fontsize=12)
    axes[0, 0].set_ylabel('Layer', fontsize=12)
    
    # Plot 2: Top circuits
    top_circuits = aggregated.nlargest(5, "impact_difference")
    y_pos = np.arange(len(top_circuits))
    axes[0, 1].barh(
        y_pos,
        top_circuits["impact_difference"],
        color='purple',
        alpha=0.7
    )
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(
        [f"L{l}H{h}" for l, h in zip(top_circuits["layer"], top_circuits["head"])]
    )
    axes[0, 1].set_xlabel('Impact Difference', fontsize=12)
    axes[0, 1].set_title('Top Critical Circuits (Averaged)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Impact by layer
    layer_impact = aggregated.groupby("layer")["impact_difference"].mean()
    axes[1, 0].bar(layer_impact.index, layer_impact.values, color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Layer', fontsize=12)
    axes[1, 0].set_ylabel('Mean Impact Difference', fontsize=12)
    axes[1, 0].set_title('Average Impact by Layer', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Impact by head
    head_impact = aggregated.groupby("head")["impact_difference"].mean()
    axes[1, 1].bar(head_impact.index, head_impact.values, color='coral', alpha=0.7)
    axes[1, 1].set_xlabel('Head', fontsize=12)
    axes[1, 1].set_ylabel('Mean Impact Difference', fontsize=12)
    axes[1, 1].set_title('Average Impact by Head', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = output_dir / "circuit_analysis_aggregated.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved aggregated circuit analysis to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate interpretability results across variants')
    parser.add_argument('--results-file', type=str, default='results/outputs/analysis_results.csv',
                       help='Path to analysis results CSV')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                       help='Directory to save aggregated figures')
    parser.add_argument('--recompute-attention', action='store_true',
                       help='Recompute full attention arrays for aggregation (slower)')
    
    args = parser.parse_args()
    
    # Load results
    results_df = pd.read_csv(args.results_file)
    logger.info(f"Loaded results for {len(results_df)} variants")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate
    aggregate_ablation_results(results_df, output_dir)
    aggregate_circuit_results(results_df, output_dir)
    aggregate_attention_results(results_df, output_dir, args.recompute_attention)
    
    logger.info("\n✓ Aggregation complete!")


if __name__ == "__main__":
    main()

