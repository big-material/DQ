
from DCC import *
from Utils import *
from Plots import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import defaultdict

init_plotting()

dataset2name = {
    "Bala_classification_dataset.csv": "Bala Classification",
    "Bala_regression_dataset.csv": "Bala Regression",
    "bandgap.csv": "Bandgap",
    "BMDS_data.csv": "BMDS",
    "Crystal_structure.csv": "Crystal Structure",
    "Glass.csv": "Glass",
    "PUE.csv": "PUE",
}


from enum import Enum
from sklearn.datasets import make_classification


class PerturbationType(Enum):
    Deletion = "Deletion"
    AdditionLinear = "AdditionLinear"
    AdditionRand = "AdditionRand"
    ReplacementLinear = "ReplacementLinear"
    ReplacementRand = "ReplacementRand"


def generate_random_according_df(
    n: int, m: int, df: pd.DataFrame, task_type=None, target_col=None
):
    cols = df.columns
    new_data = []
    for i in range(n):
        new_row = []
        for j in range(m):
            new_row.append(random.uniform(df[cols[j]].min(), df[cols[j]].max()))
        new_data.append(new_row)
    new_df = pd.DataFrame(new_data, columns=cols)
    if task_type == "classification":
        if target_col is not None:
            new_df[target_col] = np.random.choice(
                df[target_col].unique(), size=n, replace=True
            )
    return new_df


def generate_linear_according_df(
    n: int, m: int, df: pd.DataFrame, task_type=None, target_col=None
):
    if task_type == "classification":
        # Generate a classification dataset
        X, y = make_classification(
            n_samples=n,
            n_features=m - 1,
            n_classes=len(df[target_col].unique()),
            n_informative= (m - 1) // 2,
        )[:2]
    else:
        X, y = make_regression(
            n_samples=n, n_features=m - 1, noise=0.1
        )[:2]

    X_cols = [x for x in df.columns if x != target_col]
    df_linear = pd.DataFrame(X, columns=X_cols)
    df_linear[target_col] = y
    # normalize the data to the range of df
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_linear[col] = (df_linear[col] - df_linear[col].min()) / (
            df_linear[col].max() - df_linear[col].min()
        ) * (max_val - min_val) + min_val

    return df_linear


def dcc_exp(
    data: pd.DataFrame,
    ptb_type: PerturbationType,
    task_type,
    target_col,
    ratio=0.1,
    eps=0.04,
    corr_func=pearson_matrix,
):
    eps /= 2
    n, m = data.shape
    if ptb_type == PerturbationType.Deletion:
        perturbed_data = data.sample(frac=1 - ratio)
    elif ptb_type == PerturbationType.AdditionLinear:
        df_linear = generate_linear_according_df(
            n // 20, m, data, task_type, target_col
        )
        perturbed_data = pd.concat([data, df_linear], ignore_index=True)
    elif ptb_type == PerturbationType.AdditionRand:
        df_rand = generate_random_according_df(n // 20, m, data, task_type, target_col)
        perturbed_data = pd.concat([data, df_rand], ignore_index=True)
    elif ptb_type == PerturbationType.ReplacementLinear:
        df_linear = generate_linear_according_df(
            n // 20, m, data, task_type, target_col
        )
        perturbed_data = random_replace_rows(data, df_linear)
    elif ptb_type == PerturbationType.ReplacementRand:
        df_rand = generate_random_according_df(n // 20, m, data, task_type, target_col)
        perturbed_data = random_replace_rows(data, df_rand)
    else:
        raise ValueError(f"Unknown perturbation type: {ptb_type}")

    return dcc_diff_features(data, perturbed_data, target_col, eps=eps, corr_func=corr_func), perturbed_data


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from joblib import Parallel, delayed
import shap

dataset_dir = Path("processed_data")
datasets = list(dataset_dir.glob("*.csv"))
dataset_config = {
    "Bala_classification_dataset.csv": {
        "target_col": "Formability",
        "type": "classification",
    },
    "Bala_regression_dataset.csv": {
        "target_col": "Ferroelectric_Tc_in_Kelvin",
        "type": "regression",
    },
    "bandgap.csv": {
        "target_col": "target",
        "type": "regression",
    },
    "BMDS_data.csv": {
        "target_col": "soc_bandgap",
        "type": "regression",
    },
    "Crystal_structure.csv": {
        "target_col": "Lowest distortion",
        "type": "classification",
    },
    "Glass.csv": {
        "target_col": "Type of glass",
        "type": "classification",
    },
    "PUE.csv": {
        "target_col": "logYM",
        "type": "regression",
    },
}
def get_shap_values(df, target_col, type):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if type == "regression":
        model = RandomForestRegressor(n_estimators=500, random_state=42)
    elif type == "classification":
        model = RandomForestClassifier(n_estimators=500, random_state=42)
    else:
        raise ValueError("type must be either 'regression' or 'classification'")
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    if type == "classification":
        # For classification, shap_values is a list of arrays, one for each class
        # We take the mean absolute value across all classes
        mean_abs_shap = np.mean(np.mean(np.abs(shap_values), axis=0), axis=1)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    # print(f"Mean absolute SHAP values: {mean_abs_shap}")
    feats2shap = {col: mean_abs_shap[i] for i, col in enumerate(X.columns)}
    return feats2shap


def get_corr_info(df, corr_func, target_col):
    corr_matrix = corr_func(df)
    feats2corr = {}
    for col in df.columns:
        if col == target_col:
            continue
        feats2corr[col] = corr_matrix.loc[target_col, col]
    return feats2corr


import pickle
from joblib import Parallel, delayed
import logging

# Define correlation functions and perturbation ratios
correlation_functions = [
    pearson_matrix,
    spearman_matrix,
    kendall_matrix,
    mutual_info_matrix,
    js_corr_matrix,
    wd_corr_matrix,
    xi_matrix,
    dcor_matrix,
]

# Configure logging for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_dataset(dataset_path, ptb_ratio, corr_func):
    """Process a single dataset with given perturbation ratio and correlation function."""
    dataset_results = {}
    
    try:
        df = pd.read_csv(dataset_path)
        dataset_name = dataset_path.name
        target_col = dataset_config[dataset_name]["target_col"]
        task_type = dataset_config[dataset_name]["type"]
        
        # Calculate original SHAP values
        original_shap = get_shap_values(df, target_col, task_type)
        
        dataset_results["ori_feats2shap"] = original_shap
        dataset_results["name"] = dataset2name[dataset_name]
        dataset_results["type"] = task_type
        dataset_results["target_col"] = target_col
        
        # Initialize perturbation result containers
        for ptb_type in PerturbationType:
            dataset_results[ptb_type.value] = {}
        
        # Process each perturbation type
        for ptb_type in PerturbationType:
            dcc_result, modified_df = dcc_exp(
                df, ptb_type, task_type, target_col, 
                ratio=ptb_ratio, eps=0.04, corr_func=corr_func
            )
            
            perturbed_shap = get_shap_values(modified_df, target_col, task_type)
            
            
            dataset_results[ptb_type.value]["dcc"] = dcc_result
            dataset_results[ptb_type.value]["feats2shap"] = perturbed_shap
            for corr_func in correlation_functions:
                correlation_info = get_corr_info(modified_df, corr_func, target_col)
                dataset_results[ptb_type.value]["feats2corr"][corr_func.__name__] = correlation_info
            
        return dataset_name, dataset_results
        
    except Exception as e:
        logging.error(f"Error processing {dataset_path.name}: {str(e)}")
        return dataset_path.name, None

def process_ratio_corr_combination(ptb_ratio, corr_func):
    """Process all datasets for a specific ratio and correlation function combination."""
    logging.info(f"Processing ratio: {ptb_ratio}, correlation: {corr_func.__name__}")
    
    dataset_paths = list(Path("processed_data").glob("*.csv"))
    
    # Parallel processing of datasets
    dataset_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_single_dataset)(dataset_path, ptb_ratio, corr_func) 
        for dataset_path in dataset_paths
    )
    
    # Consolidate results
    consolidated_results = defaultdict(dict)
    for dataset_name, result_data in dataset_results:
        if result_data is not None:
            consolidated_results[dataset_name] = result_data
    
    # Save consolidated results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"results_{ptb_ratio}_{corr_func.__name__}.pkl"
    
    with open(output_file, "wb") as file_handle:
        pickle.dump(dict(consolidated_results), file_handle)
    
    logging.info(f"Saved results to {output_file}")
    return output_file



if __name__ == "__main__":
    perturbation_ratios = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]

    # Create all ratio-correlation combinations
    ratio_corr_combinations = [
        (ratio, corr_func) for ratio in perturbation_ratios 
        for corr_func in correlation_functions
    ]

    # Parallel processing of ratio-correlation combinations
    # Using fewer jobs for outer loop to avoid overwhelming the system
    processed_files = Parallel(n_jobs=-1, verbose=2)(
        delayed(process_ratio_corr_combination)(ratio, corr_func)
        for ratio, corr_func in ratio_corr_combinations
    )

    logging.info(f"All processing complete. Generated {len(processed_files)} result files.")