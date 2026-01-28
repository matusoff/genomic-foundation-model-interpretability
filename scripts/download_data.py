"""
Script to download and prepare the genomic-FM dataset.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import VariantDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Download and prepare genomic-FM dataset')
    parser.add_argument('--repo-url', default='https://github.com/bowang-lab/genomic-FM',
                       help='GitHub repository URL')
    parser.add_argument('--variant-file', type=str, default=None,
                       help='Specific variant file to use (optional)')
    parser.add_argument('--context-size', type=int, default=512,
                       help='Sequence context size')
    parser.add_argument('--reference-genome', type=str, default=None,
                       help='Path to reference genome FASTA file (optional)')
    parser.add_argument('--n-variants', type=int, default=None,
                       help='Limit number of variants to process')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Downloading and Preparing Genomic-FM Dataset")
    logger.info("="*60)
    
    # Initialize data loader
    loader = VariantDataLoader()
    
    # Download/clone repository
    try:
        dataset_path = loader.download_genomic_fm_dataset(repo_url=args.repo_url)
        logger.info(f"Dataset available at: {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("\nYou can manually clone the repository:")
        logger.info(f"  git clone {args.repo_url} data/genomic-FM")
        logger.info("\nOr download variant files directly from the repository.")
        return
    
    # Load variants
    logger.info("\nLoading variants...")
    try:
        variants_df = loader.load_genomic_fm_variants(
            dataset_path=str(dataset_path),
            variant_file=args.variant_file
        )
        
        if args.n_variants:
            variants_df = variants_df.head(args.n_variants)
            logger.info(f"Limited to {args.n_variants} variants")
        
        logger.info(f"Loaded {len(variants_df)} variants")
        logger.info(f"Columns: {list(variants_df.columns)}")
        
        # Prepare sequences
        logger.info("\nPreparing sequence contexts...")
        dataset_df = loader.prepare_variant_dataset_from_genomic_fm(
            variants_df,
            context_size=args.context_size,
            reference_genome_path=args.reference_genome
        )
        
        logger.info(f"\n✓ Dataset prepared successfully!")
        logger.info(f"  Total variants: {len(dataset_df)}")
        logger.info(f"  Output file: data/variants/variants_with_sequences.csv")
        
        if 'pathogenic' in dataset_df.columns:
            n_pathogenic = dataset_df['pathogenic'].sum()
            n_benign = (~dataset_df['pathogenic']).sum()
            logger.info(f"  Pathogenic: {n_pathogenic}, Benign: {n_benign}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

