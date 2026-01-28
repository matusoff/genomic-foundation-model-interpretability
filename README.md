# Genomic Foundation Model Interpretability Analysis

Mechanistic interpretability analysis of genomic foundation models to understand how they predict functional impact of genetic variants.

## Overview

This project applies interpretability techniques to genomic foundation models to:
- Understand how models interpret genetic variants
- Identify key sequence features and internal model components
- Relate findings to known biological mechanisms

**Model:** Nucleotide Transformer v2 (100M parameters, 22 layers, 16 attention heads)  
**Dataset:** Genetic variants from [genomic-FM](https://github.com/bowang-lab/genomic-FM) (NeurIPS 2024)

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/omatu/genomic-foundation-model-interpretability.git
cd genomic-foundation-model-interpretability

# Create conda environment (recommended)
conda create -n genomic_interpretability python=3.9
conda activate genomic_interpretability

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download and prepare variant dataset
python scripts/download_data.py --n-variants 50

# This creates: data/variants/variants_with_sequences.csv
```

### 3. Run Analysis

```bash
# Run complete analysis pipeline
python run_analysis.py \
  --variants data/variants/variants_with_sequences.csv \
  --n-variants 10 \
  --skip-sae \
  --model-name "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

# Results saved to: results/
```

### 4. Generate Aggregated Results

```bash
# Create averaged figures across all variants
python scripts/aggregate_results.py

# Generates:
# - results/figures/ablation_analysis_aggregated.png
# - results/figures/circuit_analysis_aggregated.png
```

### 5. Generate PDF Report

```bash
# Create 2-4 page PDF report
python scripts/generate_report.py

# Output: Genomic_Model_Interpretability_Report.pdf
```

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_analysis.py                    # Main analysis script
│
├── src/                               # Source code
│   ├── model_loader_simple.py        # Model loading (with fallbacks)
│   ├── data_loader.py                # Variant data preparation
│   ├── attention_analysis.py         # Attention visualization
│   ├── activation_patching.py        # Position ablation
│   ├── circuit_analysis.py           # Attention head ablation
│   └── sparse_autoencoder.py         # SAE analysis
│
├── scripts/
│   ├── download_data.py              # Download genomic-FM dataset
│   ├── aggregate_results.py          # Aggregate across variants
│   └── generate_report.py            # Generate PDF report
│
├── data/                              # Data directory
│   └── variants/                      # Variant datasets
│       └── variants_with_sequences.csv
│
├── results/                            # Analysis results
│   ├── figures/                        # Visualizations
│   │   ├── ablation_analysis_aggregated.png
│   │   ├── circuit_analysis_aggregated.png
│   │   └── variant_*_attention/        # Per-variant attention plots
│   └── outputs/
│       └── analysis_results.csv       # Summary statistics
│
└── docs/                               # Documentation
    ├── INTERPRETABILITY_METHODS_EXPLAINED.md
    ├── REPORT_GUIDANCE.md
    └── BUGS_FIXED.md
```

## Interpretability Methods

### 1. Attention Visualization
- **What:** Extracts attention weights from all layers/heads
- **Shows:** Where the model "looks" when processing sequences
- **Output:** Heatmaps comparing ref vs alt attention patterns

### 2. Position Ablation
- **What:** Systematically masks sequence positions
- **How:** Masks 5-nucleotide windows at 40 positions (±100 nt around variant)
- **Shows:** Which sequence regions are most critical
- **Output:** Impact plots showing critical positions

### 3. Circuit Analysis
- **What:** Ablates individual attention heads
- **How:** Tests 80 layer-head combinations (last 10 layers × 8 heads)
- **Shows:** Which internal model components matter
- **Output:** Heatmaps and top critical circuits

### 4. Sparse Autoencoder (Optional)
- **What:** Discovers latent interpretable features
- **Note:** Slower, can be skipped with `--skip-sae`

See [INTERPRETABILITY_METHODS_EXPLAINED.md](INTERPRETABILITY_METHODS_EXPLAINED.md) for detailed explanations.

## Command-Line Options

### `run_analysis.py`

```bash
python run_analysis.py \
  --variants PATH              # Variant CSV file
  --n-variants N              # Number of variants to analyze
  --skip-sae                  # Skip SAE analysis (faster)
  --model-name MODEL          # Specific HuggingFace model
  --circuit-layers N          # Layers to test (default: 10)
  --circuit-heads N           # Heads to test (default: 8)
  --results-dir DIR           # Results directory
```

### `scripts/download_data.py`

```bash
python scripts/download_data.py \
  --n-variants N              # Number of variants
  --context-size N            # Sequence context size (default: 512)
  --reference-genome PATH     # Optional: reference genome FASTA
```

## Data Format

Input variant CSV should have columns:
- `variant_id`: Unique identifier (e.g., "chr7:117559593:G>A")
- `chr`: Chromosome
- `pos`: Position (1-based)
- `ref`: Reference allele
- `alt`: Alternate allele
- `pathogenic`: Boolean (optional)

The script automatically prepares sequences if not present.

## Model Information

- **Model:** InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
- **Architecture:** 22 layers, 16 attention heads, 100M parameters
- **Tokenization:** 6-mer k-mers (~6 nucleotides per token)
- **Download:** Automatic on first run (requires internet)
- **Cache:** Models cached in `models/` directory

## Results Interpretation

### Ablation Analysis
- **High impact positions** = Critical sequence regions
- **Differential impact (Alt - Ref)** = Variant-specific sensitivity

### Circuit Analysis
- **Deeper layers** (18-21) typically more important
- **Top circuits** = Layer-head combinations critical for variant discrimination

### Attention Analysis
- **High attention to variant** = Model recognizes the change
- **Pattern differences** = How model's "reading" changes

## Troubleshooting

### Model Download Issues
```bash
# Force offline mode (use cached models only)
set HF_HUB_OFFLINE=1  # Windows
export HF_HUB_OFFLINE=1  # Linux/Mac
```

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce `--n-variants`
- Use `--skip-sae`
- Reduce `--circuit-layers` and `--circuit-heads`

See [INSTALL_TROUBLESHOOTING.md](INSTALL_TROUBLESHOOTING.md) for more.

## Citation

If you use this code, please cite:

- **Nucleotide Transformer v2:** Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics" (2023)
- **Genomic-FM Dataset:** Wang et al., "Genomic-FM: A Foundation Model for DNA Sequences" (NeurIPS 2024)
- **Repository:** https://github.com/<your-username>/<repository-name>

## License

[Specify your license]

## Contact

For questions: vallijah.subasri@uhn.ca

## Acknowledgments

- Model: InstaDeepAI Nucleotide Transformer v2
- Dataset: bowang-lab/genomic-FM
- Framework: HuggingFace Transformers, PyTorch
