"""
Data loading and preparation for genetic variant analysis.
Handles downloading variant datasets and preparing sequence contexts.
Supports the genomic-FM dataset from bowang-lab (NeurIPS 2024).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import requests
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
import shutil
import subprocess
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VariantDataLoader:
    """Loader for genetic variant datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.variants_dir = self.data_dir / "variants"
        self.sequences_dir = self.data_dir / "sequences"
        
        # Create directories
        self.variants_dir.mkdir(parents=True, exist_ok=True)
        self.sequences_dir.mkdir(parents=True, exist_ok=True)
    
    def create_example_variants(self, n_variants: int = 50) -> pd.DataFrame:
        """
        Create example variant dataset for testing.
        In production, this would download from the actual database.
        
        Args:
            n_variants: Number of variants to generate
            
        Returns:
            DataFrame with variant information
        """
        logger.info(f"Creating example variant dataset with {n_variants} variants")
        
        # Example pathogenic and benign variants
        variants = []
        
        # Pathogenic variants (examples from known disease genes)
        pathogenic_examples = [
            {"chr": "chr7", "pos": 117559593, "ref": "G", "alt": "A", "gene": "CFTR", "pathogenic": True},
            {"chr": "chr17", "pos": 43093829, "ref": "G", "alt": "A", "gene": "TP53", "pathogenic": True},
            {"chr": "chr11", "pos": 5248232, "ref": "C", "alt": "T", "gene": "HBB", "pathogenic": True},
        ]
        
        # Generate variants
        for i in range(n_variants):
            if i < len(pathogenic_examples):
                variant = pathogenic_examples[i].copy()
            else:
                # Generate synthetic variants
                is_pathogenic = np.random.random() > 0.5
                variant = {
                    "chr": f"chr{np.random.choice([1, 7, 11, 17])}",
                    "pos": np.random.randint(1000000, 250000000),
                    "ref": np.random.choice(["A", "T", "G", "C"]),
                    "alt": np.random.choice(["A", "T", "G", "C"]),
                    "gene": f"GENE_{i}",
                    "pathogenic": is_pathogenic
                }
            
            variant["variant_id"] = f"{variant['chr']}:{variant['pos']}:{variant['ref']}>{variant['alt']}"
            variants.append(variant)
        
        df = pd.DataFrame(variants)
        df["label"] = df["pathogenic"].astype(int)
        
        # Save to CSV
        output_path = self.variants_dir / "example_variants.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved variants to {output_path}")
        
        return df
    
    def prepare_sequence_context(
        self,
        variant: Dict,
        context_size: int = 512,
        reference_genome: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Prepare sequence context around a variant.
        
        Args:
            variant: Dictionary with chr, pos, ref, alt
            context_size: Size of context window (total, split around variant)
            reference_genome: Path to reference genome FASTA (optional)
            
        Returns:
            Tuple of (sequence_context, variant_position_in_context)
        """
        # For this example, generate synthetic sequence context
        # In production, would fetch from reference genome
        
        half_context = context_size // 2
        
        # Generate random sequence context
        bases = ["A", "T", "G", "C"]
        upstream = "".join(np.random.choice(bases, half_context))
        downstream = "".join(np.random.choice(bases, half_context))
        
        # Create reference sequence
        ref_seq = upstream + variant["ref"] + downstream
        
        # Create alternate sequence
        alt_seq = upstream + variant["alt"] + downstream
        
        variant_pos = half_context
        
        return {
            "ref_sequence": ref_seq,
            "alt_sequence": alt_seq,
            "variant_position": variant_pos,
            "context_size": context_size
        }
    
    def prepare_variant_dataset(
        self,
        variants_df: pd.DataFrame,
        context_size: int = 512
    ) -> pd.DataFrame:
        """
        Prepare full dataset with sequence contexts.
        
        Args:
            variants_df: DataFrame with variant information
            context_size: Size of context window
            
        Returns:
            DataFrame with added sequence contexts
        """
        logger.info(f"Preparing sequence contexts for {len(variants_df)} variants")
        
        sequences = []
        for idx, row in variants_df.iterrows():
            variant = {
                "chr": row["chr"],
                "pos": row["pos"],
                "ref": row["ref"],
                "alt": row["alt"]
            }
            
            seq_context = self.prepare_sequence_context(variant, context_size)
            
            sequences.append({
                "variant_id": row["variant_id"],
                "ref_sequence": seq_context["ref_sequence"],
                "alt_sequence": seq_context["alt_sequence"],
                "variant_position": seq_context["variant_position"],
                "label": row["label"],
                "pathogenic": row["pathogenic"]
            })
        
        seq_df = pd.DataFrame(sequences)
        
        # Merge with original variants
        result_df = variants_df.merge(seq_df, on="variant_id", how="left")
        
        # Save
        output_path = self.variants_dir / "variants_with_sequences.csv"
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")
        
        return result_df
    
    def download_genomic_fm_dataset(
        self,
        repo_url: str = "https://github.com/bowang-lab/genomic-FM",
        clone_dir: Optional[str] = None
    ) -> Path:
        """
        Download/clone the genomic-FM dataset from GitHub.
        
        Args:
            repo_url: GitHub repository URL
            clone_dir: Directory to clone into (default: data/genomic-FM)
            
        Returns:
            Path to cloned repository
        """
        if clone_dir is None:
            clone_dir = self.data_dir / "genomic-FM"
        else:
            clone_dir = Path(clone_dir)
        
        if clone_dir.exists() and (clone_dir / ".git").exists():
            logger.info(f"Repository already exists at {clone_dir}")
            return clone_dir
        
        logger.info(f"Cloning genomic-FM repository from {repo_url}")
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(clone_dir)],
                check=True,
                capture_output=True
            )
            logger.info(f"Successfully cloned repository to {clone_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            logger.info("You can manually clone the repo or download the data files")
            raise
        except FileNotFoundError:
            logger.warning("Git not found. Please manually clone the repository:")
            logger.warning(f"  git clone {repo_url} {clone_dir}")
            raise
        
        return clone_dir
    
    def load_genomic_fm_variants(
        self,
        dataset_path: Optional[str] = None,
        variant_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load variants from the genomic-FM dataset.
        
        Args:
            dataset_path: Path to genomic-FM repository
            variant_file: Specific variant file to load
            
        Returns:
            DataFrame with variants
        """
        if dataset_path is None:
            dataset_path = self.data_dir / "genomic-FM"
        else:
            dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            logger.info("Dataset not found. Downloading...")
            dataset_path = self.download_genomic_fm_dataset()
        
        # Look for variant files (common formats: .tsv, .csv, .vcf)
        if variant_file is None:
            # Try to find variant files
            possible_files = list(dataset_path.rglob("*.tsv")) + \
                           list(dataset_path.rglob("*.csv")) + \
                           list(dataset_path.rglob("*.vcf"))
            
            if not possible_files:
                logger.warning("No variant files found in dataset. Using example data.")
                return self.create_example_variants()
            
            variant_file = possible_files[0]
            logger.info(f"Found variant file: {variant_file}")
        else:
            variant_file = dataset_path / variant_file
        
        # Load variant file
        try:
            if variant_file.suffix == '.vcf':
                df = self._load_vcf(variant_file)
            else:
                # Try CSV/TSV
                df = pd.read_csv(variant_file, sep='\t' if variant_file.suffix == '.tsv' else ',')
            
            logger.info(f"Loaded {len(df)} variants from {variant_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading variant file: {e}")
            logger.info("Falling back to example dataset")
            return self.create_example_variants()
    
    def _load_vcf(self, vcf_path: Path) -> pd.DataFrame:
        """Load VCF file and convert to DataFrame."""
        variants = []
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    variants.append({
                        'chr': parts[0],
                        'pos': int(parts[1]),
                        'ref': parts[3],
                        'alt': parts[4].split(',')[0],  # Take first alternate
                        'variant_id': f"{parts[0]}:{parts[1]}:{parts[3]}>{parts[4].split(',')[0]}"
                    })
        return pd.DataFrame(variants)
    
    def fetch_sequence_context_from_genome(
        self,
        chrom: str,
        pos: int,
        context_size: int = 512,
        reference_genome_path: Optional[str] = None
    ) -> str:
        """
        Fetch sequence context from reference genome.
        
        Args:
            chrom: Chromosome (e.g., 'chr1' or '1')
            pos: Position (1-based)
            context_size: Total context size (split around position)
            reference_genome_path: Path to reference genome FASTA
            
        Returns:
            Sequence context string
        """
        # If no reference genome provided, generate synthetic context
        # In production, would use pyfaidx or pysam to fetch from FASTA
        if reference_genome_path is None or not Path(reference_genome_path).exists():
            logger.warning("Reference genome not provided. Using synthetic sequence.")
            bases = ["A", "T", "G", "C"]
            half_context = context_size // 2
            return "".join(np.random.choice(bases, context_size))
        
        # Use pyfaidx to fetch from reference genome
        try:
            import pyfaidx
            genome = pyfaidx.Fasta(reference_genome_path)
            
            # Normalize chromosome name
            chrom_clean = chrom.replace('chr', '')
            if chrom_clean in genome.keys():
                chrom_key = chrom_clean
            elif chrom in genome.keys():
                chrom_key = chrom
            else:
                logger.warning(f"Chromosome {chrom} not found in reference genome")
                return self._generate_synthetic_sequence(context_size)
            
            half_context = context_size // 2
            start = max(1, pos - half_context)
            end = pos + half_context
            
            sequence = str(genome[chrom_key][start-1:end])
            return sequence.upper()
        except ImportError:
            logger.warning("pyfaidx not available. Using synthetic sequence.")
            return self._generate_synthetic_sequence(context_size)
        except Exception as e:
            logger.error(f"Error fetching sequence: {e}")
            return self._generate_synthetic_sequence(context_size)
    
    def _generate_synthetic_sequence(self, length: int) -> str:
        """Generate synthetic DNA sequence."""
        bases = ["A", "T", "G", "C"]
        return "".join(np.random.choice(bases, length))
    
    def prepare_variant_dataset_from_genomic_fm(
        self,
        variants_df: pd.DataFrame,
        context_size: int = 512,
        reference_genome_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare dataset with sequence contexts from genomic-FM variants.
        
        Args:
            variants_df: DataFrame with variants (must have chr, pos, ref, alt)
            context_size: Size of context window
            reference_genome_path: Path to reference genome FASTA
            
        Returns:
            DataFrame with added sequence contexts
        """
        logger.info(f"Preparing sequence contexts for {len(variants_df)} variants")
        
        sequences = []
        for idx, row in variants_df.iterrows():
            chrom = str(row.get('chr', row.get('chromosome', 'chr1')))
            pos = int(row.get('pos', row.get('position', 1)))
            ref = str(row.get('ref', 'A'))
            alt = str(row.get('alt', 'T'))
            
            # Fetch sequence context
            full_context = self.fetch_sequence_context_from_genome(
                chrom, pos, context_size, reference_genome_path
            )
            
            # Find variant position in context
            half_context = context_size // 2
            variant_pos_in_context = half_context
            
            # Ensure ref allele is correctly placed in the context
            # (important when using synthetic sequences)
            ref_sequence = (
                full_context[:variant_pos_in_context] +
                ref +
                full_context[variant_pos_in_context + len(ref):]
            )
            
            # Create alternate sequence
            alt_sequence = (
                full_context[:variant_pos_in_context] +
                alt +
                full_context[variant_pos_in_context + len(ref):]
            )
            
            sequences.append({
                "variant_id": row.get('variant_id', f"{chrom}:{pos}:{ref}>{alt}"),
                "ref_sequence": ref_sequence,
                "alt_sequence": alt_sequence,
                "variant_position": variant_pos_in_context,
                "chr": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt
            })
        
        seq_df = pd.DataFrame(sequences)
        
        # Merge with original variants
        if 'variant_id' in variants_df.columns:
            result_df = variants_df.merge(seq_df, on="variant_id", how="left", suffixes=('', '_seq'))
        else:
            result_df = pd.concat([variants_df, seq_df], axis=1)
        
        # Add label if available
        if 'pathogenic' not in result_df.columns and 'label' not in result_df.columns:
            # Try to infer from other columns
            if 'clinical_significance' in result_df.columns:
                result_df['pathogenic'] = result_df['clinical_significance'].str.contains(
                    'pathogenic|disease', case=False, na=False
                )
            else:
                result_df['pathogenic'] = False  # Default to benign if unknown
        
        if 'label' not in result_df.columns:
            result_df['label'] = result_df['pathogenic'].astype(int)
        
        # Save
        output_path = self.variants_dir / "variants_with_sequences.csv"
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")
        
        return result_df
    
    def load_variants(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load variant dataset.
        
        Args:
            filepath: Path to variant CSV file
            
        Returns:
            DataFrame with variants
        """
        if filepath is None:
            filepath = self.variants_dir / "variants_with_sequences.csv"
        
        if not Path(filepath).exists():
            logger.warning(f"File not found: {filepath}")
            logger.info("Attempting to load from genomic-FM dataset...")
            try:
                variants_df = self.load_genomic_fm_variants()
                return self.prepare_variant_dataset_from_genomic_fm(variants_df)
            except Exception as e:
                logger.warning(f"Could not load genomic-FM dataset: {e}")
                logger.info("Creating example dataset...")
                variants_df = self.create_example_variants()
                return self.prepare_variant_dataset(variants_df)
        
        return pd.read_csv(filepath)


if __name__ == "__main__":
    # Test data loading
    loader = VariantDataLoader()
    
    # Create example dataset
    variants_df = loader.create_example_variants(n_variants=20)
    print(f"Created {len(variants_df)} variants")
    print(variants_df.head())
    
    # Prepare sequences
    dataset_df = loader.prepare_variant_dataset(variants_df, context_size=256)
    print(f"\nDataset with sequences:")
    print(dataset_df[["variant_id", "ref_sequence", "alt_sequence", "label"]].head())

