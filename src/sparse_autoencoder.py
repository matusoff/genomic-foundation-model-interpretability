"""
Sparse Autoencoder (SAE) analysis for discovering interpretable features.
Trains SAE on model activations to find meaningful biological patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import logging

try:
    from .model_loader import GenomicModelLoader
except ImportError:
    from .model_loader_simple import SimpleGenomicModelLoader as GenomicModelLoader
from .data_loader import VariantDataLoader

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for feature discovery."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coef: float = 0.01
    ):
        """
        Initialize sparse autoencoder.
        
        Args:
            input_dim: Input dimension (activation size)
            hidden_dim: Hidden dimension (number of features)
            sparsity_coef: Sparsity coefficient for L1 regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def loss(self, x, x_reconstructed, encoded):
        """
        Compute sparse autoencoder loss.
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed input
            encoded: Encoded representation
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_reconstructed, x)
        
        # Sparsity loss (L1 on activations)
        sparsity_loss = self.sparsity_coef * encoded.abs().mean()
        
        total_loss = recon_loss + sparsity_loss
        
        return total_loss, recon_loss, sparsity_loss


class SparseAutoencoderAnalyzer:
    """Analyzer using sparse autoencoders for interpretability."""
    
    def __init__(
        self,
        model_loader: GenomicModelLoader,
        results_dir: str = "results"
    ):
        """
        Initialize SAE analyzer.
        
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
        self.sae = None
    
    def extract_activations(
        self,
        sequences: List[str],
        layer: int = -1,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Extract activations from model for given sequences.
        
        Args:
            sequences: List of DNA sequences
            layer: Layer to extract from (-1 for last layer)
            max_length: Maximum sequence length
            
        Returns:
            Array of activations [num_sequences, seq_len, hidden_dim]
        """
        logger.info(f"Extracting activations from {len(sequences)} sequences")
        
        all_activations = []
        
        for seq in tqdm(sequences, desc="Extracting activations"):
            outputs = self.model_loader.forward_with_attentions(seq, max_length)
            
            # Handle different output structures (ESM vs standard)
            activations = None
            
            # Try to get from hidden_states first
            if "hidden_states" in outputs and outputs["hidden_states"] is not None:
                hidden_states = outputs["hidden_states"]
                if isinstance(hidden_states, tuple):
                    # ESM models return tuple
                    if layer == -1:
                        activations = hidden_states[-1]
                    else:
                        activations = hidden_states[layer] if layer < len(hidden_states) else hidden_states[-1]
                elif hasattr(hidden_states, '__getitem__'):
                    # Standard format
                    if layer == -1:
                        activations = hidden_states[-1]
                    else:
                        activations = hidden_states[layer] if layer < len(hidden_states) else hidden_states[-1]
            
            # Fallback to last_hidden_state
            if activations is None and "last_hidden_state" in outputs and outputs["last_hidden_state"] is not None:
                activations = outputs["last_hidden_state"]
            
            if activations is None:
                logger.warning(f"Could not extract activations for sequence, skipping")
                continue
            
            # Remove batch dimension and move to CPU
            if isinstance(activations, torch.Tensor):
                if activations.dim() == 3:
                    activations = activations.squeeze(0)  # Remove batch dimension
                activations = activations.cpu().numpy()
            else:
                activations = np.array(activations)
                if activations.ndim == 3:
                    activations = activations[0]  # Remove batch dimension
            
            all_activations.append(activations)
        
        # Pad to same length
        max_len = max(a.shape[0] for a in all_activations)
        padded_activations = []
        
        for act in all_activations:
            pad_len = max_len - act.shape[0]
            if pad_len > 0:
                act = np.pad(act, ((0, pad_len), (0, 0)), mode='constant')
            padded_activations.append(act)
        
        return np.array(padded_activations)
    
    def train_sae(
        self,
        activations: np.ndarray,
        hidden_dim: int = 512,
        sparsity_coef: float = 0.01,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> Dict:
        """
        Train sparse autoencoder on activations.
        
        Args:
            activations: Activation array [num_sequences, seq_len, hidden_dim]
            hidden_dim: Number of features to discover
            sparsity_coef: Sparsity coefficient
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Dictionary with training history and model
        """
        logger.info(f"Training SAE with {hidden_dim} features")
        
        # Flatten activations: [num_sequences * seq_len, hidden_dim]
        num_seqs, seq_len, act_dim = activations.shape
        activations_flat = activations.reshape(-1, act_dim)
        
        # Normalize
        scaler = StandardScaler()
        activations_scaled = scaler.fit_transform(activations_flat)
        
        # Convert to tensor
        activations_tensor = torch.FloatTensor(activations_scaled).to(self.device)
        
        # Initialize SAE
        self.sae = SparseAutoencoder(
            input_dim=act_dim,
            hidden_dim=hidden_dim,
            sparsity_coef=sparsity_coef
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=lr)
        
        # Training
        losses = []
        recon_losses = []
        sparsity_losses = []
        
        dataset = torch.utils.data.TensorDataset(activations_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in tqdm(range(epochs), desc="Training SAE"):
            epoch_losses = []
            epoch_recon = []
            epoch_sparse = []
            
            for batch in dataloader:
                x = batch[0]
                
                optimizer.zero_grad()
                x_recon, encoded = self.sae(x)
                loss, recon_loss, sparse_loss = self.sae.loss(x, x_recon, encoded)
                
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_recon.append(recon_loss.item())
                epoch_sparse.append(sparse_loss.item())
            
            losses.append(np.mean(epoch_losses))
            recon_losses.append(np.mean(epoch_recon))
            sparsity_losses.append(np.mean(epoch_sparse))
        
        return {
            "sae": self.sae,
            "scaler": scaler,
            "losses": losses,
            "recon_losses": recon_losses,
            "sparsity_losses": sparsity_losses
        }
    
    def analyze_features(
        self,
        ref_sequences: List[str],
        alt_sequences: List[str],
        layer: int = -1
    ) -> Dict:
        """
        Analyze SAE features for variant sequences.
        
        Args:
            ref_sequences: Reference sequences
            alt_sequences: Alternate sequences
            layer: Layer to analyze
            
        Returns:
            Dictionary with feature analysis
        """
        if self.sae is None:
            raise ValueError("SAE not trained. Call train_sae first.")
        
        # Extract activations
        ref_activations = self.extract_activations(ref_sequences, layer)
        alt_activations = self.extract_activations(alt_sequences, layer)
        
        # Encode with SAE
        ref_activations_flat = ref_activations.reshape(-1, ref_activations.shape[-1])
        alt_activations_flat = alt_activations.reshape(-1, alt_activations.shape[-1])
        
        # Normalize (using same scaler from training)
        # Note: In practice, should use the scaler from training
        ref_encoded = self.sae.encoder(
            torch.FloatTensor(ref_activations_flat).to(self.device)
        ).detach().cpu().numpy()
        
        alt_encoded = self.sae.encoder(
            torch.FloatTensor(alt_activations_flat).to(self.device)
        ).detach().cpu().numpy()
        
        # Analyze feature differences
        feature_diff = alt_encoded.mean(axis=0) - ref_encoded.mean(axis=0)
        feature_std_ref = ref_encoded.std(axis=0)
        feature_std_alt = alt_encoded.std(axis=0)
        
        # Find most discriminative features
        top_features = np.argsort(np.abs(feature_diff))[-20:][::-1]
        
        return {
            "ref_features": ref_encoded,
            "alt_features": alt_encoded,
            "feature_differences": feature_diff,
            "top_features": top_features,
            "feature_stats": {
                "ref_mean": ref_encoded.mean(axis=0),
                "alt_mean": alt_encoded.mean(axis=0),
                "ref_std": feature_std_ref,
                "alt_std": feature_std_alt
            }
        }
    
    def visualize_features(
        self,
        feature_analysis: Dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize SAE feature analysis.
        
        Args:
            feature_analysis: Results from analyze_features
            save_path: Path to save figure
        """
        top_features = feature_analysis["top_features"][:10]
        feature_diff = feature_analysis["feature_differences"]
        ref_mean = feature_analysis["feature_stats"]["ref_mean"]
        alt_mean = feature_analysis["feature_stats"]["alt_mean"]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Top feature differences
        feature_indices = top_features
        x_pos = np.arange(len(feature_indices))
        
        axes[0].barh(x_pos, feature_diff[feature_indices], color='purple', alpha=0.7)
        axes[0].set_yticks(x_pos)
        axes[0].set_yticklabels([f"Feature {i}" for i in feature_indices])
        axes[0].set_xlabel('Feature Activation Difference (Alt - Ref)', fontsize=12)
        axes[0].set_title('Top Discriminative SAE Features', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Feature activation comparison
        axes[1].bar(x_pos - 0.2, ref_mean[feature_indices], 0.4, label='Reference', alpha=0.7)
        axes[1].bar(x_pos + 0.2, alt_mean[feature_indices], 0.4, label='Alternate', alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f"F{i}" for i in feature_indices], rotation=45)
        axes[1].set_ylabel('Mean Feature Activation', fontsize=12)
        axes[1].set_title('Feature Activation Comparison', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature visualization to {save_path}")
        
        return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test SAE
    model_loader = GenomicModelLoader()
    analyzer = SparseAutoencoderAnalyzer(model_loader)
    
    # Test sequences
    sequences = ["ATGC" * 50] * 10
    
    # Extract activations
    activations = analyzer.extract_activations(sequences, layer=-1)
    print(f"Activations shape: {activations.shape}")
    
    # Train SAE
    training_results = analyzer.train_sae(activations, hidden_dim=256, epochs=10)
    print(f"Final loss: {training_results['losses'][-1]:.4f}")

