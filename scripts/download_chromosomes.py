"""
Download only the chromosomes needed for analysis (much smaller than full genome).
Downloads individual chromosome files from UCSC (~50-200MB each instead of 3-9GB total).
"""

import requests
import gzip
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chromosomes needed based on your variants
CHROMOSOMES_NEEDED = ['chr1', 'chr7', 'chr11', 'chr17']

# UCSC download URL
UCSC_BASE_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/'

def download_chromosome(chr_name: str, output_dir: Path) -> Path:
    """Download a single chromosome file from UCSC."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # UCSC uses format: chr1.fa.gz, chr7.fa.gz, etc.
    filename = f"{chr_name}.fa.gz"
    url = f"{UCSC_BASE_URL}{filename}"
    
    output_path = output_dir / filename
    decompressed_path = output_dir / f"{chr_name}.fa"
    
    # Skip if already downloaded
    if decompressed_path.exists():
        logger.info(f"{chr_name} already exists at {decompressed_path}")
        return decompressed_path
    
    logger.info(f"Downloading {chr_name} from {url}...")
    logger.info(f"This may take a few minutes (file size: ~50-200MB)...")
    
    try:
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) == 0:  # Print every 10MB
                            logger.info(f"  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
        
        logger.info(f"Downloaded {chr_name}.fa.gz ({downloaded / 1024 / 1024:.1f}MB)")
        
        # Decompress
        logger.info(f"Decompressing {chr_name}...")
        with gzip.open(output_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove compressed file to save space
        output_path.unlink()
        
        logger.info(f"✓ {chr_name} ready at {decompressed_path}")
        return decompressed_path
        
    except Exception as e:
        logger.error(f"Error downloading {chr_name}: {e}")
        raise

def combine_chromosomes(chr_files: list, output_path: Path):
    """Combine multiple chromosome files into one FASTA file."""
    logger.info(f"Combining chromosomes into {output_path}...")
    
    with open(output_path, 'wb') as outfile:
        for chr_file in chr_files:
            logger.info(f"  Adding {chr_file.name}...")
            with open(chr_file, 'rb') as infile:
                outfile.write(infile.read())
    
    logger.info(f"✓ Combined genome ready at {output_path}")

def main():
    """Download only the chromosomes needed for analysis."""
    logger.info("="*60)
    logger.info("Downloading Chromosomes for Variant Analysis")
    logger.info("="*60)
    logger.info(f"Chromosomes needed: {', '.join(CHROMOSOMES_NEEDED)}")
    logger.info(f"Total: 4 chromosomes (~200-800MB instead of 3-9GB!)")
    logger.info("")
    
    # Create output directory
    output_dir = Path("data/reference_genome")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each chromosome
    chr_files = []
    for chr_name in CHROMOSOMES_NEEDED:
        try:
            chr_file = download_chromosome(chr_name, output_dir)
            chr_files.append(chr_file)
        except Exception as e:
            logger.error(f"Failed to download {chr_name}: {e}")
            logger.info("You can manually download from:")
            logger.info(f"  {UCSC_BASE_URL}{chr_name}.fa.gz")
            return
    
    # Combine into one file (optional, but makes it easier to use)
    combined_path = output_dir / "hg38_subset.fa"
    if not combined_path.exists():
        combine_chromosomes(chr_files, combined_path)
    
    logger.info("")
    logger.info("="*60)
    logger.info("✓ Download Complete!")
    logger.info("="*60)
    logger.info(f"Reference genome subset saved to: {combined_path}")
    logger.info(f"Total size: {combined_path.stat().st_size / 1024 / 1024:.1f}MB")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Install pyfaidx: pip install pyfaidx")
    logger.info("2. Re-prepare data: python scripts/download_data.py --reference-genome data/reference_genome/hg38_subset.fa --n-variants 10")
    logger.info("3. Re-run analysis: python run_analysis.py --variants data/variants/variants_with_sequences.csv --n-variants 10 --skip-sae")

if __name__ == "__main__":
    main()

