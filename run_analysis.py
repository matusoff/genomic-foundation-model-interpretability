"""
Main script to run complete interpretability analysis.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

try:
    from src.model_loader import GenomicModelLoader
except:
    from src.model_loader_simple import SimpleGenomicModelLoader as GenomicModelLoader
from src.data_loader import VariantDataLoader
from src.attention_analysis import AttentionAnalyzer
from src.activation_patching import ActivationPatcher
from src.sparse_autoencoder import SparseAutoencoderAnalyzer
from src.circuit_analysis import CircuitAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run genomic model interpretability analysis')
    parser.add_argument('--variants', type=str, default='data/variants/example_variants.csv',
                       help='Path to variant dataset')
    parser.add_argument('--n-variants', type=int, default=10,
                       help='Number of variants to analyze')
    parser.add_argument('--context-size', type=int, default=512,
                       help='Sequence context size')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--skip-sae', action='store_true',
                       help='Skip sparse autoencoder analysis (faster)')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Specific model to use (e.g., InstaDeepAI/nucleotide-transformer-v2-500m-multi-species). If not specified, will try DNA models in order.')
    parser.add_argument('--circuit-layers', type=int, default=10,
                       help='Number of layers to test in circuit analysis (default: 10 = last 10 layers)')
    parser.add_argument('--circuit-heads', type=int, default=8,
                       help='Number of heads to test in circuit analysis (default: 8)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start analysis from this variant index (0-based). Useful for re-running specific variants.')
    
    args = parser.parse_args()
    
    logger.info("Starting interpretability analysis")
    logger.info(f"Variants file: {args.variants}")
    logger.info(f"Number of variants: {args.n_variants}")
    
    # Initialize components
    logger.info("Loading model...")
    # Always use SimpleGenomicModelLoader which has DNA model fallbacks
    from src.model_loader_simple import SimpleGenomicModelLoader
    model_loader = SimpleGenomicModelLoader(model_name=args.model_name)
    model_loader.load()
    model_info = model_loader.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Layers: {model_info['num_layers']}, Heads: {model_info['num_heads']}")
    
    # Verify we're using a DNA model, not a protein model
    if 'esm' in model_info['model_name'].lower():
        logger.warning("⚠️  WARNING: Using ESM (protein) model instead of DNA model!")
        logger.warning("⚠️  The test requires DNA/genomic models (e.g., Nucleotide Transformer v2)")
        logger.warning("⚠️  To fix: Delete cached model or use --model-name to specify DNA model")
    
    # Load data
    logger.info("Loading variant data...")
    data_loader = VariantDataLoader()
    
    if not Path(args.variants).exists():
        logger.info("Creating example dataset...")
        variants_df = data_loader.create_example_variants(n_variants=args.n_variants * 2)
        dataset_df = data_loader.prepare_variant_dataset(variants_df, context_size=args.context_size)
    else:
        dataset_df = data_loader.load_variants(args.variants)
    
    # Select subset for analysis
    if args.start_index > 0:
        logger.info(f"Starting from variant index {args.start_index}")
        analysis_df = dataset_df.iloc[args.start_index:args.start_index + args.n_variants]
    else:
        analysis_df = dataset_df.head(args.n_variants)
    logger.info(f"Analyzing {len(analysis_df)} variants (indices {args.start_index} to {args.start_index + len(analysis_df) - 1})")
    
    # Initialize analyzers
    attention_analyzer = AttentionAnalyzer(model_loader, args.results_dir)
    patcher = ActivationPatcher(model_loader, args.results_dir)
    circuit_analyzer = CircuitAnalyzer(model_loader, args.results_dir)
    
    if not args.skip_sae:
        sae_analyzer = SparseAutoencoderAnalyzer(model_loader, args.results_dir)
    
    # Run analyses
    all_results = []
    
    for idx, row in analysis_df.iterrows():
        logger.info(f"\nAnalyzing variant {idx+1}/{len(analysis_df)}: {row['variant_id']}")
        
        ref_seq = row['ref_sequence']
        alt_seq = row['alt_sequence']
        variant_pos = row['variant_position']
        is_pathogenic = row['pathogenic']
        
        variant_results = {
            "variant_id": row['variant_id'],
            "pathogenic": is_pathogenic,
            "variant_position": variant_pos
        }
        
        # 1. Attention analysis
        logger.info("  Running attention analysis...")
        try:
            attn_results = attention_analyzer.compare_variant_attentions(
                ref_seq,
                alt_seq,
                variant_pos,
                save_dir=str(Path(args.results_dir) / "figures" / f"variant_{idx}_attention")
            )
            variant_results["attention"] = {
                "variant_attention_diff": attn_results["variant_attention_diff"].mean().item(),
                "max_attention_diff": attn_results["variant_attention_diff"].max().item()
            }
        except Exception as e:
            logger.warning(f"  Attention analysis failed: {e}")
            variant_results["attention"] = None
        
        # 2. Activation patching
        logger.info("  Running activation patching...")
        try:
            critical_positions = patcher.analyze_critical_positions(
                ref_seq,
                alt_seq,
                variant_pos,
                top_k=5
            )
            variant_results["critical_positions"] = critical_positions["top_critical_positions"].to_dict('records')
        except Exception as e:
            logger.warning(f"  Activation patching failed: {e}")
            variant_results["critical_positions"] = None
        
        # 3. Circuit analysis
        logger.info("  Running circuit analysis...")
        try:
            # Speed up: sample layers and heads (default: 80 combinations instead of 352)
            circuits = circuit_analyzer.find_critical_circuits(
                ref_seq,
                alt_seq,
                top_k=5,
                sample_layers=args.circuit_layers,
                sample_heads=args.circuit_heads
            )
            variant_results["circuits"] = circuits["top_circuits"].to_dict('records')
        except Exception as e:
            logger.warning(f"  Circuit analysis failed: {e}")
            variant_results["circuits"] = None
        
        # 4. Sparse autoencoder (optional, slower)
        if not args.skip_sae:
            logger.info("  Running SAE analysis...")
            try:
                # Extract activations
                ref_activations = sae_analyzer.extract_activations([ref_seq], layer=-1)
                alt_activations = sae_analyzer.extract_activations([alt_seq], layer=-1)
                
                # Train SAE (use multiple sequences for better training)
                all_activations = np.concatenate([ref_activations, alt_activations], axis=0)
                training_results = sae_analyzer.train_sae(
                    all_activations,
                    hidden_dim=128,
                    epochs=20
                )
                
                # Analyze features
                feature_analysis = sae_analyzer.analyze_features(
                    [ref_seq],
                    [alt_seq],
                    layer=-1
                )
                
                sae_analyzer.visualize_features(
                    feature_analysis,
                    save_path=str(Path(args.results_dir) / "figures" / f"variant_{idx}_sae.png")
                )
                
                variant_results["sae"] = {
                    "top_features": feature_analysis["top_features"][:5].tolist(),
                    "max_feature_diff": feature_analysis["feature_differences"].max()
                }
            except Exception as e:
                logger.warning(f"  SAE analysis failed: {e}")
                variant_results["sae"] = None
        
        all_results.append(variant_results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = Path(args.results_dir) / "outputs" / "analysis_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")
    
    # Summary statistics
    logger.info("\n" + "="*50)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*50)
    
    successful_attn = sum(1 for r in all_results if r.get("attention") is not None)
    successful_patch = sum(1 for r in all_results if r.get("critical_positions") is not None)
    successful_circuit = sum(1 for r in all_results if r.get("circuits") is not None)
    
    logger.info(f"Variants analyzed: {len(all_results)}")
    logger.info(f"Successful attention analyses: {successful_attn}")
    logger.info(f"Successful activation patching: {successful_patch}")
    logger.info(f"Successful circuit analyses: {successful_circuit}")
    
    if not args.skip_sae:
        successful_sae = sum(1 for r in all_results if r.get("sae") is not None)
        logger.info(f"Successful SAE analyses: {successful_sae}")
    
    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()

