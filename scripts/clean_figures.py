"""
Clean up old timestamped figure files, keeping only aggregated and latest versions.
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_figures(figures_dir: str = "results/figures", keep_aggregated: bool = True):
    """
    Remove timestamped figure files, keeping only:
    - aggregated figures
    - latest versions (ablation_analysis.png, circuit_analysis.png)
    - variant attention folders
    
    Args:
        figures_dir: Directory containing figures
        keep_aggregated: Keep aggregated figures
    """
    fig_path = Path(figures_dir)
    
    if not fig_path.exists():
        logger.warning(f"Figures directory not found: {figures_dir}")
        return
    
    # Files to keep
    keep_patterns = [
        "*_aggregated.png",
        "ablation_analysis.png",  # Latest version
        "circuit_analysis.png",   # Latest version
    ]
    
    # Count files to delete
    timestamped_ablation = list(fig_path.glob("ablation_analysis_*.png"))
    timestamped_circuit = list(fig_path.glob("circuit_analysis_*.png"))
    
    # Filter out files to keep
    to_delete = []
    for f in timestamped_ablation + timestamped_circuit:
        if f.name not in ["ablation_analysis.png", "circuit_analysis.png"]:
            if not (keep_aggregated and "_aggregated" in f.name):
                to_delete.append(f)
    
    logger.info(f"Found {len(to_delete)} timestamped files to delete")
    logger.info(f"Keeping aggregated and latest versions")
    
    if len(to_delete) == 0:
        logger.info("No files to clean")
        return
    
    # Delete files
    deleted = 0
    for f in to_delete:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete {f.name}: {e}")
    
    logger.info(f"✓ Deleted {deleted} files")
    logger.info(f"Kept: ablation_analysis.png, circuit_analysis.png, and aggregated figures")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Clean up timestamped figure files')
    parser.add_argument('--figures-dir', type=str, default='results/figures',
                       help='Figures directory')
    parser.add_argument('--keep-aggregated', action='store_true', default=True,
                       help='Keep aggregated figures')
    
    args = parser.parse_args()
    clean_figures(args.figures_dir, args.keep_aggregated)

