# Setup Guide: Genomic Foundation Model Interpretability

## Step-by-Step Setup Instructions

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download and Prepare Data

The test requires using the **genomic-FM dataset** from the NeurIPS 2024 paper.

#### Option A: Automatic Download (Recommended)

```bash
# Download the dataset from GitHub
python scripts/download_data.py
```

This will:
1. Clone the repository: https://github.com/bowang-lab/genomic-FM
2. Load variant data files
3. Prepare sequence contexts for each variant
4. Save to `data/variants/variants_with_sequences.csv`

#### Option B: Manual Download

If automatic download fails:

```bash
# Clone the repository manually
git clone https://github.com/bowang-lab/genomic-FM data/genomic-FM

# Then run the data preparation
python scripts/download_data.py --variant-file path/to/variant/file.tsv
```

#### Option C: Use Your Own Variant Dataset

If you have your own variant dataset:

1. Format it as CSV/TSV with columns: `chr`, `pos`, `ref`, `alt`, `pathogenic` (optional)
2. Place it in `data/variants/`
3. Run: `python scripts/download_data.py --variant-file data/variants/your_file.csv`

### Step 3: Prepare Sequence Contexts

Sequence contexts are automatically prepared during data download. However, if you want to use a reference genome:

```bash
# Download GRCh38 reference genome (if needed)
# You can get it from: https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.26/

# Then specify it:
python scripts/download_data.py --reference-genome /path/to/GRCh38.fa
```

**Note**: If no reference genome is provided, synthetic sequences will be used for demonstration. For real analysis, use the actual reference genome.

### Step 4: Run the Analysis

```bash
# Basic analysis (fast)
python run_analysis.py --n-variants 10

# Full analysis with all methods
python run_analysis.py --n-variants 20 --skip-sae false

# Use specific variant file
python run_analysis.py --variants data/variants/variants_with_sequences.csv --n-variants 15
```

### Step 5: Generate Report

```bash
# Generate PDF report
python scripts/generate_report.py --results-dir results --output report/report.pdf
```

## What Each Step Does

### Data Preparation (`scripts/download_data.py`)
- Downloads/clones the genomic-FM repository
- Loads variant data (VCF, TSV, or CSV format)
- Extracts sequence contexts around each variant
- Creates reference and alternate sequences
- Saves prepared dataset

### Analysis (`run_analysis.py`)
- Loads the model (Nucleotide Transformer v2)
- For each variant, runs:
  - **Attention Analysis**: Compares attention patterns
  - **Activation Patching**: Finds critical positions
  - **Circuit Analysis**: Identifies important layer-head combinations
  - **Sparse Autoencoder**: Discovers interpretable features (optional)
- Generates visualizations
- Saves results to `results/`

### Report Generation (`scripts/generate_report.py`)
- Compiles results into PDF
- Includes key findings and visualizations
- Summarizes biological interpretations

## Troubleshooting

### Issue: Git not found
**Solution**: Install Git or manually download the repository

### Issue: Model download fails
**Solution**: Check internet connection. The model will be downloaded from Hugging Face on first use.

### Issue: Out of memory
**Solution**: 
- Reduce `--n-variants`
- Use `--skip-sae` to skip sparse autoencoder (most memory-intensive)
- Use smaller `--context-size`

### Issue: No variant files found
**Solution**: 
- Check that the repository was cloned correctly
- Manually specify variant file: `--variant-file path/to/file.tsv`
- Use example data for testing

### Issue: Reference genome not found
**Solution**: 
- The code will use synthetic sequences if no reference genome is provided
- For real analysis, download GRCh38 from NCBI
- Or use the UCSC Genome Browser downloads

## Expected Outputs

After running the analysis, you should have:

1. **Prepared Dataset**: `data/variants/variants_with_sequences.csv`
2. **Analysis Results**: `results/outputs/analysis_results.csv`
3. **Visualizations**: `results/figures/*.png`
4. **Report**: `report/report.pdf`

## Next Steps

1. Review the generated visualizations in `results/figures/`
2. Check the analysis results in `results/outputs/analysis_results.csv`
3. Read the generated report: `report/report.pdf`
4. Customize the analysis for your specific research questions

## Questions?

If you encounter issues:
1. Check the logs for error messages
2. Ensure all dependencies are installed
3. Verify the dataset was downloaded correctly
4. Try with a smaller number of variants first

