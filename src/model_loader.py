"""
Model loading utilities for genomic foundation models.
Supports Nucleotide Transformer v2 and other Hugging Face models.
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenomicModelLoader:
    """Loader for genomic foundation models."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: Hugging Face model identifier (None for auto-select)
            device: 'cuda' or 'cpu' (auto-detected if None)
            cache_dir: Directory to cache models
        """
        # Try alternative models if primary fails
        if model_name is None:
            # Try smaller, more accessible models first
            self.model_name = "facebook/esm2_t6_8M_UR50D"  # Much smaller, easier to download
        else:
            self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or "models"
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        
    def load(self) -> Tuple[AutoModel, AutoTokenizer]:
        """Load model and tokenizer."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            # Try AutoModelForMaskedLM first (for ESM-based models)
            try:
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    output_attentions=True,
                    output_hidden_states=True,
                    trust_remote_code=True
                )
            except Exception:
                # Fallback to AutoModel
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    output_attentions=True,
                    output_hidden_states=True,
                    trust_remote_code=True
                )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        
        return self.model, self.tokenizer
    
    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 1024,
        return_tensors: str = "pt"
    ) -> dict:
        """
        Encode DNA sequence for model input.
        
        Args:
            sequence: DNA sequence string (A, T, G, C)
            max_length: Maximum sequence length
            return_tensors: 'pt' for PyTorch tensors
            
        Returns:
            Tokenized input dictionary
        """
        if self.tokenizer is None:
            self.load()
        
        # Tokenize sequence
        encoded = self.tokenizer(
            sequence,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        return encoded
    
    def forward_with_attentions(
        self,
        sequence: str,
        max_length: int = 1024
    ) -> dict:
        """
        Run forward pass and return outputs with attentions.
        
        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with outputs, attentions, and hidden states
        """
        if self.model is None:
            self.load()
        
        encoded = self.encode_sequence(sequence, max_length)
        
        with torch.no_grad():
            outputs = self.model(**encoded, output_attentions=True, output_hidden_states=True)
        
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states,
            "input_ids": encoded["input_ids"]
        }
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        if self.model is None:
            self.load()
        
        return {
            "model_name": self.model_name,
            "num_layers": self.model.config.num_hidden_layers,
            "num_heads": self.model.config.num_attention_heads,
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": self.model.config.vocab_size,
            "device": self.device
        }


if __name__ == "__main__":
    # Test model loading
    loader = GenomicModelLoader()
    model, tokenizer = loader.load()
    info = loader.get_model_info()
    print("Model Info:", info)
    
    # Test encoding
    test_seq = "ATGCATGCATGC"
    encoded = loader.encode_sequence(test_seq)
    print(f"Encoded sequence shape: {encoded['input_ids'].shape}")
    
    # Test forward pass
    outputs = loader.forward_with_attentions(test_seq)
    print(f"Hidden states shape: {outputs['last_hidden_state'].shape}")
    print(f"Number of attention layers: {len(outputs['attentions'])}")

