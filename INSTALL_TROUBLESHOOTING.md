# Installation Troubleshooting

## Issue: pysam Installation Fails on Windows

**Problem**: `pysam` requires compilation and C libraries, which can fail on Windows.

**Solution**: `pysam` and `pyfaidx` are **optional** dependencies. They're only needed if needed to fetch sequences from a reference genome FASTA file.

### Quick Fix

The updated `requirements.txt` no longer includes `pysam` and `pyfaidx`. Install the core dependencies:

```bash
pip install -r requirements.txt
```

### If Reference Genome Support Needed

If you want to use a reference genome FASTA file, install these separately:

**Option 1: Use Conda (Recommended for Windows)**
```bash
conda install -c bioconda pyfaidx pysam
```

**Option 2: Install pyfaidx only (pysam is harder on Windows)**
```bash
pip install pyfaidx
```

**Note**: The code will work fine without these - it will use synthetic sequences for demonstration, or you can provide pre-extracted sequences.

## Other Common Issues

### Issue: CUDA/GPU not available
**Solution**: The code works on CPU, just slower. Install CPU-only PyTorch if needed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Out of memory during model download
**Solution**: The model is ~2GB. Ensure:
- At least 4GB free disk space
- Stable internet connection
- Model downloads automatically on first use

### Issue: Git not found (for dataset download)
**Solution**: 
- Install Git from https://git-scm.com/download/win
- Or manually download the dataset from GitHub

## Verify Installation

After installation, verify everything works:

```bash
python quick_start.py
```

This will check:
- All required packages are installed
- Model can be loaded
- Data can be prepared

