"""
Evaluate Nucleotide Transformer model performance on variant classification.

This script:
1. Extracts embeddings from ref and alt sequences
2. Trains a simple classifier on top of embeddings
3. Evaluates with AUC, accuracy, F1 score
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader_simple import SimpleGenomicModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_embeddings(model_loader, sequences, batch_size=4):
    """Extract embeddings for a list of sequences."""
    model_loader.model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
            batch_seqs = sequences[i:i+batch_size]
            
            # Tokenize
            encoded = model_loader.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            # Forward pass
            outputs = model_loader.model(**encoded)
            
            # Get embeddings (mean pooling over sequence length)
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden = outputs.hidden_states[-1]
            else:
                hidden = outputs.logits
            
            # Mean pooling: average over sequence length (excluding padding)
            attention_mask = encoded['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
            
            embeddings.append(pooled.cpu().numpy())
    
    return np.vstack(embeddings)


def prepare_features(ref_embeddings, alt_embeddings, method='difference'):
    """Prepare features from ref and alt embeddings.
    
    Args:
        ref_embeddings: Reference sequence embeddings
        alt_embeddings: Alternate sequence embeddings
        method: 'difference', 'concatenate', or 'both'
    """
    if method == 'difference':
        return alt_embeddings - ref_embeddings
    elif method == 'concatenate':
        return np.concatenate([ref_embeddings, alt_embeddings], axis=1)
    elif method == 'both':
        diff = alt_embeddings - ref_embeddings
        concat = np.concatenate([ref_embeddings, alt_embeddings], axis=1)
        return np.concatenate([diff, concat], axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_model(
    variants_file: str = "data/variants/variants_with_sequences.csv",
    model_name: str = None,
    test_size: float = 0.3,
    classifier_type: str = "logistic",
    feature_method: str = "difference",
    output_file: str = "results/outputs/model_performance.csv"
):
    """Evaluate model performance on variant classification."""
    
    logger.info("="*60)
    logger.info("Model Performance Evaluation")
    logger.info("="*60)
    
    # Load variants
    logger.info(f"Loading variants from {variants_file}...")
    variants_df = pd.read_csv(variants_file)
    
    # Filter variants with valid labels
    if 'pathogenic' in variants_df.columns:
        variants_df = variants_df[variants_df['pathogenic'].notna()]
        labels = variants_df['pathogenic'].astype(int).values
    elif 'label' in variants_df.columns:
        variants_df = variants_df[variants_df['label'].notna()]
        labels = variants_df['label'].astype(int).values
    else:
        logger.error("No 'pathogenic' or 'label' column found in variants file")
        return
    
    logger.info(f"Loaded {len(variants_df)} variants")
    logger.info(f"  Pathogenic: {labels.sum()}, Benign: {(~labels.astype(bool)).sum()}")
    
    # Check for valid sequences
    if 'ref_sequence' not in variants_df.columns or 'alt_sequence' not in variants_df.columns:
        logger.error("Missing 'ref_sequence' or 'alt_sequence' columns")
        return
    
    # Filter out variants with invalid sequences (e.g., all N's)
    valid_mask = (
        variants_df['ref_sequence'].str.len() > 100 &
        variants_df['alt_sequence'].str.len() > 100 &
        (~variants_df['ref_sequence'].str.contains('^N+$', regex=True)) &
        (~variants_df['alt_sequence'].str.contains('^N+$', regex=True))
    )
    variants_df = variants_df[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask]
    
    logger.info(f"After filtering: {len(variants_df)} variants")
    
    if len(variants_df) < 10:
        logger.warning("Too few variants for meaningful evaluation. Need at least 10.")
        return
    
    # Load model
    logger.info("Loading model...")
    model_loader = SimpleGenomicModelLoader(model_name=model_name)
    model_loader.load()
    model_info = model_loader.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    
    # Extract embeddings
    logger.info("Extracting reference sequence embeddings...")
    ref_embeddings = extract_embeddings(model_loader, variants_df['ref_sequence'].tolist())
    
    logger.info("Extracting alternate sequence embeddings...")
    alt_embeddings = extract_embeddings(model_loader, variants_df['alt_sequence'].tolist())
    
    # Prepare features
    logger.info(f"Preparing features using method: {feature_method}...")
    X = prepare_features(ref_embeddings, alt_embeddings, method=feature_method)
    y = labels
    
    logger.info(f"Feature shape: {X.shape}")
    
    # Split data
    if len(variants_df) < 20:
        # Too small for train/test split, use all data for training
        logger.warning("="*60)
        logger.warning("WARNING: Dataset too small for train/test split.")
        logger.warning("Metrics will be computed on training data (optimistic).")
        logger.warning("For proper evaluation, need 20+ variants.")
        logger.warning("="*60)
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train classifier
    logger.info(f"Training {classifier_type} classifier...")
    if classifier_type == "logistic":
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif classifier_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else clf.predict_proba(X_test)[:, 0]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
    
    # AUC (only if we have both classes in test set)
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = np.nan
        logger.warning("Only one class in test set, cannot compute AUC")
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    if not np.isnan(auc):
        logger.info(f"AUC-ROC:   {auc:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Benign', 'Pathogenic']))
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['Accuracy', 'F1_Score', 'AUC_ROC'],
        'value': [accuracy, f1, auc if not np.isnan(auc) else None],
        'classifier': [classifier_type] * 3,
        'feature_method': [feature_method] * 3,
        'n_train': [len(X_train)] * 3,
        'n_test': [len(X_test)] * 3
    })
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'classifier': classifier_type,
        'feature_method': feature_method
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance on variant classification')
    parser.add_argument('--variants', type=str, default='data/variants/variants_with_sequences.csv',
                        help='Path to variants CSV file')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name (default: auto-detect)')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test set size (default: 0.3)')
    parser.add_argument('--classifier', type=str, default='logistic',
                        choices=['logistic', 'random_forest'],
                        help='Classifier type (default: logistic)')
    parser.add_argument('--feature-method', type=str, default='difference',
                        choices=['difference', 'concatenate', 'both'],
                        help='Feature preparation method (default: difference)')
    parser.add_argument('--output', type=str, default='results/outputs/model_performance.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    evaluate_model(
        variants_file=args.variants,
        model_name=args.model_name,
        test_size=args.test_size,
        classifier_type=args.classifier,
        feature_method=args.feature_method,
        output_file=args.output
    )

