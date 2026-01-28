"""
Simplified model loader with fallback options and better error handling.
Uses smaller, more accessible models if the primary model fails.
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from typing import Optional, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGenomicModelLoader:
    """Simplified loader with fallback models."""
    
    # List of DNA/genomic models to try, in order of preference
    # NOTE: These are DNA sequence models, NOT protein models
    MODEL_OPTIONS = [
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",  # Smaller Nucleotide Transformer (DNA) - recommended (already cached)
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",  # Medium Nucleotide Transformer (DNA)
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",  # Large Nucleotide Transformer v2 (DNA)
        # Alternative DNA models (if Nucleotide Transformer fails):
        # "zhihan1996/DNABERT-2-117M",  # DNABERT-2 (DNA)
        # "LongSafari/hyenadna-tiny-1k",  # HyenaDNA (DNA)
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize model loader with fallback options.
        
        Args:
            model_name: Specific model to use (None for auto-select)
            device: 'cuda' or 'cpu' (auto-detected if None)
            cache_dir: Directory to cache models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or "models"
        
        if model_name:
            self.model_options = [model_name]
        else:
            self.model_options = self.MODEL_OPTIONS.copy()
        
        self.model_name = None
        self.tokenizer = None
        self.model = None
        
    def load(self) -> Tuple[AutoModel, AutoTokenizer]:
        """Load model and tokenizer, trying fallback options if needed."""
        if self.model is not None:
            return self.model, self.tokenizer
        
        last_error = None
        
        for model_name in self.model_options:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                self.model_name = model_name
                
                # Try to load tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        local_files_only=True  # Use only cached files, don't check online
                    )
                except Exception as e:
                    logger.warning(f"Tokenizer load failed for {model_name}: {e}")
                    # Try without trust_remote_code
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=self.cache_dir
                    )
                
                # Try to load model
                # Note: timeout is not a valid parameter for model initialization
                # It's only used in the HTTP request, which happens before this
                try:
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        model_name,
                        cache_dir=self.cache_dir,
                        output_attentions=True,
                        output_hidden_states=True,
                        trust_remote_code=True,
                        local_files_only=True  # Use only cached files
                    )
                except Exception as e1:
                    logger.warning(f"AutoModelForMaskedLM failed, trying AutoModel: {e1}")
                    try:
                        self.model = AutoModel.from_pretrained(
                            model_name,
                            cache_dir=self.cache_dir,
                            output_attentions=True,
                            output_hidden_states=True,
                            trust_remote_code=True,
                            local_files_only=True  # Use only cached files
                        )
                    except Exception as e2:
                        raise e2
                
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"✓ Successfully loaded model: {model_name}")
                return self.model, self.tokenizer
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                last_error = e
                continue
        
        # If all models failed, raise error with helpful message
        raise RuntimeError(
            f"Failed to load any model. Tried: {self.model_options}\n"
            f"Last error: {last_error}\n"
            f"Possible solutions:\n"
            f"1. Check internet connection\n"
            f"2. Try manual download: huggingface-cli download <model_name>\n"
            f"3. Use a different model by specifying model_name parameter"
        )
    
    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 1024,
        return_tensors: str = "pt"
    ) -> dict:
        """Encode DNA sequence for model input."""
        if self.tokenizer is None:
            self.load()
        
        # For DNA models, tokenize nucleotide sequences (A, T, G, C)
        # Nucleotide Transformer expects DNA sequences directly
        encoded = self.tokenizer(
            sequence,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded
    
    def forward_with_attentions(
        self,
        sequence: str,
        max_length: int = 1024
    ) -> dict:
        """Run forward pass and return outputs with attentions."""
        if self.model is None:
            self.load()
        
        encoded = self.encode_sequence(sequence, max_length)
        
        with torch.no_grad():
            outputs = self.model(**encoded, output_attentions=True, output_hidden_states=True)
        
        # Handle different output structures (Nucleotide Transformer vs standard)
        # Some models return MaskedLMOutput which has hidden_states as tuple, not last_hidden_state
        last_hidden = None
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            last_hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            if isinstance(outputs.hidden_states, tuple) and len(outputs.hidden_states) > 0:
                last_hidden = outputs.hidden_states[-1]
            elif hasattr(outputs.hidden_states, '__getitem__') and len(outputs.hidden_states) > 0:
                last_hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, 'logits'):
            # Use logits as fallback (not ideal but works)
            last_hidden = outputs.logits
        
        attentions = None
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions
        
        hidden_states = None
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        
        return {
            "last_hidden_state": last_hidden,
            "attentions": attentions,
            "hidden_states": hidden_states,
            "input_ids": encoded["input_ids"],
            "logits": outputs.logits if hasattr(outputs, 'logits') else None
        }
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        if self.model is None:
            self.load()
        
        return {
            "model_name": self.model_name,
            "num_layers": getattr(self.model.config, 'num_hidden_layers', 'unknown'),
            "num_heads": getattr(self.model.config, 'num_attention_heads', 'unknown'),
            "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
            "vocab_size": getattr(self.model.config, 'vocab_size', 'unknown'),
            "device": self.device
        }


if __name__ == "__main__":
    # Test with fallback
    loader = SimpleGenomicModelLoader()
    try:
        model, tokenizer = loader.load()
        info = loader.get_model_info()
        print("Model Info:", info)
    except Exception as e:
        print(f"Error: {e}")

