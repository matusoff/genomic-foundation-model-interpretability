"""
General visualization utilities for interpretability analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_variant_comparison(
    results: Dict,
    variant_pos: int,
    save_path: Optional[str] = None
):
    """
    Create comprehensive comparison plot for variant analysis.
    
    Args:
        results: Dictionary with analysis results
        variant_pos: Variant position
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Attention comparison (if available)
    if "attention_comparison" in results:
        attn_data = results["attention_comparison"]
        layers = np.arange(len(attn_data["ref_mean"]))
        axes[0, 0].plot(layers, attn_data["ref_mean"], 'b-o', label='Reference', linewidth=2)
        axes[0, 0].plot(layers, attn_data["alt_mean"], 'r-o', label='Alternate', linewidth=2)
        axes[0, 0].set_xlabel('Layer', fontsize=12)
        axes[0, 0].set_ylabel('Mean Attention', fontsize=12)
        axes[0, 0].set_title('Attention to Variant Position', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Ablation impact (if available)
    if "ablation_results" in results:
        ablation_df = results["ablation_results"]
        positions = ablation_df["position"].values
        impact_diff = ablation_df["impact_difference"].values
        axes[0, 1].bar(positions, impact_diff, color='purple', alpha=0.7, width=5)
        axes[0, 1].axvline(x=variant_pos, color='green', linestyle='--', linewidth=2)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0, 1].set_xlabel('Position', fontsize=12)
        axes[0, 1].set_ylabel('Impact Difference', fontsize=12)
        axes[0, 1].set_title('Ablation Impact', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Circuit importance (if available)
    if "circuit_results" in results:
        circuit_df = results["circuit_results"]
        top_circuits = circuit_df.nlargest(10, "impact_difference")
        y_pos = np.arange(len(top_circuits))
        axes[1, 0].barh(
            y_pos,
            top_circuits["impact_difference"],
            color='steelblue',
            alpha=0.7
        )
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(
            [f"L{l}H{h}" for l, h in zip(top_circuits["layer"], top_circuits["head"])]
        )
        axes[1, 0].set_xlabel('Impact Difference', fontsize=12)
        axes[1, 0].set_title('Top Critical Circuits', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Feature importance (if available)
    if "sae_features" in results:
        feature_data = results["sae_features"]
        top_features = feature_data["top_features"][:10]
        feature_diff = feature_data["feature_differences"][top_features]
        x_pos = np.arange(len(top_features))
        axes[1, 1].bar(x_pos, feature_diff, color='coral', alpha=0.7)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([f"F{i}" for i in top_features], rotation=45)
        axes[1, 1].set_ylabel('Feature Difference', fontsize=12)
        axes[1, 1].set_title('Top SAE Features', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_figure(
    all_results: Dict,
    save_path: Optional[str] = None
):
    """
    Create summary figure with key findings.
    
    Args:
        all_results: Dictionary with all analysis results
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Add title
    fig.suptitle('Genomic Foundation Model Interpretability Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add subplots based on available results
    # This is a template - customize based on your results
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

