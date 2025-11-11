
## DQ

The GUI for the application of DCC(Dataset Correlation Convergency) framework for any datasets with one selected preditive target.

### Main Features
- Correlation matrix calculation (Pearson, Spearman, etc.)
- Analysis of the impact of missing data ratio on correlation coefficients
- Feature correlation and stability evaluation
- Streamlit visualization interface

## Usage

### Launch Streamlit visualization interface
```bash
streamlit run runner.py
```

The models can be download from the [release](https://github.com/big-material/DQ/releases/tag/v1.0) .
1. The model for predicting the performance: [rf_model_scores.pkl](https://github.com/big-material/DQ/releases/download/v1.0/rf_model_scores.pkl).
2. The model for predicting the feature importance: [rf_model_shap.pkl](https://github.com/big-material/DQ/releases/download/v1.0/rf_model_shap.pkl).

### Main APIs
- `dcc(data, ratio, eps, corr_func)`: Analyze the impact of missing data ratio on correlation coefficients

For more usage, please refer to the source code and comments.

### experiments

The details about experiments is in the dir [experiment](./experiment)

# Reference

```
DCC: a model-free frame to evaluate dataset quality
```
