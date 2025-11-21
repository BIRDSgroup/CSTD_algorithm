# CSTD Code Repository

This repository contains the implementation code for the Causal Scoring Truth Discovery (CSTD) algorithm and LLM-based inference pipelines for gene-disease causal relation.

## Overview

This repository includes:
1. **CSTD Algorithm**: An iterative algorithm for computing reliability scores of sources and causal scores for gene-disease pairs
2. **LLM Inference Pipelines**: Scripts for running causal relationship predictions using MMed-Llama-3-8B and GPT-4o models
3. **Evaluation Scripts**: Code for generating precision-recall curves, recall@K curves, and classification reports

## Repository Structure

```
CSTD_code_to_submit/
├── README.md                           # This file
├── requirements.txt                    # Python package dependencies
├── CSTD_algo_to_submit.ipynb          # Main CSTD algorithm implementation
├── LLM_Inference_AD_to_submit.ipynb   # LLM inference for Alzheimer's Disease
├── LLM_Inference_PD_to_submit.ipynb   # LLM inference for Parkinson's Disease
├── cred doc casuality with preds.csv  # Input data for CSTD algorithm
├── AD_complete_data_to_submit.csv     # Alzheimer's Disease complete data
├── PD_complete_data_to_submit.csv     # Parkinson's Disease complete data
├── LLM_inference_AD_OMIM.csv          # AD LLM inference results on OMIM data
└── LLM_inference_PD_OMIM.csv          # PD LLM inference results on OMIM data
```

## Prerequisites

### Python Version
- Python 3.8 or higher (tested with Python 3.12.7)

### Required Python Packages

Install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn tqdm openai tiktoken
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Required Packages List:
- `pandas` (>= 2.0.0)
- `numpy` (>= 1.20.0)
- `matplotlib` (>= 3.5.0)
- `seaborn` (>= 0.11.0)
- `scipy` (>= 1.7.0)
- `scikit-learn` (>= 1.0.0)
- `tqdm` (>= 4.60.0)
- `openai` (>= 1.0.0)
- `tiktoken` (>= 0.5.0)

## API Credentials Setup

**Important**: API credentials are required for running the LLM inference notebooks.

### 1. Hugging Face API (for MMed-Llama-3-8B)

The notebooks use Hugging Face Inference Endpoints. You need to:
- Obtain a Hugging Face API token
- Update the `api_key` parameter in the notebook cells that use `OpenAI` client with Hugging Face endpoint
- Update the `base_url` if using a different endpoint

### 2. Azure OpenAI API (for GPT-4o)

The notebooks use Azure OpenAI service. You need to:
- Obtain Azure OpenAI API credentials
- Update the following variables in the notebook:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_DEPLOYMENT` (deployment name)

**Note**: The notebooks contain placeholder API keys. Replace them with your own credentials before running.

## Data Files

The following data files are provided in this repository:

- `cred doc casuality with preds.csv`: Input data for the CSTD algorithm containing gene-disease pairs, source information (PMIDs), and labels
- `AD_complete_data_to_submit.csv`: Complete Alzheimer's Disease dataset
- `PD_complete_data_to_submit.csv`: Complete Parkinson's Disease dataset

### Expected Data Format

**For CSTD Algorithm** (`cred doc casuality with preds.csv`):
- Required columns: `PMID`, `id1` (gene ID), `id2` (disease ID), `label` (0 or 1), `H-Index`, `Citations`, `Year difference`

**For LLM Inference**:
- Required columns: `id1` (gene ID), `id2` (disease ID), `sentence` (abstracts containing gene-disease relationship)

## Usage Instructions

### 1. CSTD Algorithm (`CSTD_algo_to_submit.ipynb`)

This notebook implements the iterative CSTD algorithm to compute:
- Reliability scores (rs) for sources (PMIDs)
- Causal scores (vq*) for gene-disease pairs

**Steps to run:**
1. Ensure `cred doc casuality with preds.csv` is in the same directory
2. Run all cells sequentially
3. The algorithm will:
   - Initialize reliability scores based on H-Index, Citations, and Year difference
   - Iteratively update weights and compute vq* values
   - Output final reliability scores and causal values

**Output:**
- Updated dataframe with `rs` (reliability score) and `vq*` (trustworthy causal value) columns
- Visualization plots (scatter plots, correlation analysis)

### 2. LLM Inference for Alzheimer's Disease (`LLM_Inference_AD_to_submit.ipynb`)

This notebook runs LLM inference for Alzheimer's Disease gene-disease pairs.

**Steps to run:**
1. Update API credentials (Hugging Face and Azure OpenAI) in the notebook
2. Ensure data file path is correct (default: `truth_discovery/AD_OMIM_complete_data` or use `AD_complete_data_to_submit.csv`)
3. Run cells sequentially:
   - **Cell 0**: MMed-Llama-3-8B inference
   - **Cell 2**: GPT-4o inference
   - **Cells 3-21**: Data processing, evaluation, and visualization

**Output:**
- `llm_results/AD_*_llm_results.csv`: Summary results
- `llm_results/AD_*_raw_responses.jsonl`: Raw LLM responses
- `LLM_inference_AD_OMIM.csv`: Final processed results
- Precision-Recall curves
- Recall@K curves
- Classification reports

**Note**: If data files are in a different location, update the `DATA_PATH` variable in the notebook.

### 3. LLM Inference for Parkinson's Disease (`LLM_Inference_PD_to_submit.ipynb`)

This notebook runs LLM inference for Parkinson's Disease gene-disease pairs.

**Steps to run:**
1. Update API credentials (Hugging Face and Azure OpenAI) in the notebook
2. Ensure data file paths are correct:
   - `truth_discovery/PD_OMIM_complete_data` or use `PD_complete_data_to_submit.csv`
   - `truth_discovery/PD_negative_class_data.csv` (if needed)
3. Run cells sequentially:
   - **Cell 0**: MMed-Llama-3-8B inference
   - **Cell 2**: GPT-4o inference
   - **Cells 3-22**: Data processing, evaluation, and visualization

**Output:**
- `llm_results/PD_*_llm_results.csv`: Summary results
- `llm_results/PD_*_raw_responses.jsonl`: Raw LLM responses
- Precision-Recall curves
- Recall@K curves
- Classification reports

**Note**: If data files are in a different location, update the `DATA_PATH` variable in the notebook.


