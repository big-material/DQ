
## DQ

The GUI for the application of DCC(Dataset Correlation Convergency) framework for any datasets with one selected preditive target.

### Main Features
- Correlation matrix calculation (Pearson, Spearman, etc.)
- Analysis of the impact of missing data ratio on correlation coefficients
- Feature correlation and stability evaluation
- Streamlit visualization interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/big-material/DQ.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
If there is no requirements.txt, please manually install pandas, numpy, scipy, scikit-learn, streamlit, joblib, dcor, etc.

## Usage

### 1. Use as a library
```python
from DQ.DCC import dcc, dcc_feature
from DQ.Correlation import pearson_matrix, spearman_matrix
import pandas as pd

df = pd.read_csv('your_data.csv')
result = dcc(df, ratio=0.1)
```

### 2. Launch Streamlit visualization interface
```bash
streamlit run runner.py
```

### 3. Main APIs
- `dcc(data, ratio, eps, corr_func)`: Analyze the impact of missing data ratio on correlation coefficients

For more usage, please refer to the source code and comments.

# Reference

Evaluate dataset quality using correlation convergency inspired from perturbation theory. submit.

