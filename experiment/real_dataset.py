from DCC import *
from Utils import *
from Plots import *
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from enum import Enum
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from joblib import Parallel, delayed
import pickle

dataset2name = {
    "Bala_classification_dataset.csv": "Bala Classification",
    "Bala_regression_dataset.csv": "Bala Regression",
    "bandgap.csv": "Bandgap",
    "BMDS_data.csv": "BMDS",
    "Crystal_structure.csv": "Crystal Structure",
    "Glass.csv": "Glass",
    "PUE.csv": "PUE",
}


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
            n_informative=(m - 1) // 2,
        )[:2]
    else:
        X, y = make_regression(n_samples=n, n_features=m - 1, noise=0.1)[:2]

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
    corr_func=nhsic_matrix,
):
    eps /= 2
    n, m = data.shape
    if ptb_type == PerturbationType.Deletion:
        perturbed_data = data.sample(frac=1 - ratio)
    elif ptb_type == PerturbationType.AdditionLinear:
        df_linear = generate_linear_according_df(
            int(n * ratio), m, data, task_type, target_col
        )
        perturbed_data = pd.concat([data, df_linear], ignore_index=True)
    elif ptb_type == PerturbationType.AdditionRand:
        df_rand = generate_random_according_df(
            int(n * ratio), m, data, task_type, target_col
        )
        perturbed_data = pd.concat([data, df_rand], ignore_index=True)
    elif ptb_type == PerturbationType.ReplacementLinear:
        df_linear = generate_linear_according_df(
            int(n * ratio), m, data, task_type, target_col
        )
        perturbed_data = random_replace_rows(data, df_linear)
    elif ptb_type == PerturbationType.ReplacementRand:
        df_rand = generate_random_according_df(
            int(n * ratio), m, data, task_type, target_col
        )
        perturbed_data = random_replace_rows(data, df_rand)
    else:
        raise ValueError(f"Unknown perturbation type: {ptb_type}")

    return dcc_diff(data, perturbed_data, eps=eps, corr_func=corr_func), perturbed_data


def get_scores_single(df, target_col, type, repeat):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    cur_scores = []
    n_splits = 5
    if type == "classification":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if type == "regression":
            model = RandomForestRegressor(n_estimators=500, random_state=42)
        elif type == "classification":
            model = RandomForestClassifier(n_estimators=500, random_state=42)
        else:
            raise ValueError("type must be either 'regression' or 'classification'")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if type == "regression":
            cur_scores.append(r2_score(y_test, y_pred))
        elif type == "classification":
            cur_scores.append(accuracy_score(y_test, y_pred))
    return cur_scores


def get_scores(df, target_col, type):
    repeats = 10

    all_scores = Parallel(n_jobs=-1)(
        delayed(get_scores_single)(df, target_col, type, i) for i in range(repeats)
    )
    all_scores = np.array(all_scores)
    return all_scores


if __name__ == "__main__":
    cur_corr_func = nhsic_matrix  # Change this to the desired correlation function
    results = defaultdict(dict)
    dataset_dir = Path("processed_data")
    for d in dataset_dir.glob("*.csv"):
        print(d.name)
        df = pd.read_csv(d)
        target_col = dataset_config[d.name]["target_col"]
        task_type = dataset_config[d.name]["type"]
        ori_scores = get_scores(df, target_col, task_type)
        results[d.name]["ori_score"] = ori_scores
        results[d.name]["name"] = dataset2name[d.name]
        results[d.name]["type"] = task_type
        results[d.name]["target_col"] = target_col
        results[d.name][PerturbationType.Deletion.value] = {}
        results[d.name][PerturbationType.AdditionLinear.value] = {}
        results[d.name][PerturbationType.AdditionRand.value] = {}
        results[d.name][PerturbationType.ReplacementLinear.value] = {}
        results[d.name][PerturbationType.ReplacementRand.value] = {}
        print(task_type, target_col)
        for ptb_type in PerturbationType:
            print(f"Perturbation type: {ptb_type.value}")
            dcc_val, perturbed_df = dcc_exp(df, ptb_type,task_type, target_col,ratio=0.05, eps=0.04, corr_func=nhsic_matrix)
            scores = get_scores(perturbed_df, target_col, task_type)
            results[d.name][ptb_type.value]["dcc"] = dcc_val
            results[d.name][ptb_type.value]["scores"] = scores

    with open(f"results/dcc_results-{cur_corr_func.__name__}.pkl", "wb") as f:
        pickle.dump(results, f)