import pandas as pd
from .Correlation import *
import numpy as np
from sklearn.model_selection import KFold
from typing import List

def dcc(data: pd.DataFrame, ratio=0.1, eps=0.01, corr_func=pearson_matrix):
    """
    计算数据缺失比例对相关系数的影响
    :param data: pd.DataFrame, 数据
    :param ratio: float, 要删除的数据比例
    :param eps: float, 阈值
    :param corr_func: function, 相关函数
    :return : float, 相关系数的差异
    """
    if ratio == 0.0:
        return 1.0
    eps /= 2
    corr_1 = corr_func(data)  # 计算原始数据的相关系数矩阵
    sampled_data = data.sample(frac=1 - ratio, axis=0)  # 随机删除部分数据
    corr_2 = corr_func(sampled_data)  # 计算删除部分数据后的相关系数矩阵
    diff = corr_1 - corr_2  # 计算两个矩阵的差异
    return (float((diff.abs() < eps).sum().sum())) / (
        float(data.shape[1]) * float(data.shape[1])
    )

def dcc_feature(
    data: pd.DataFrame, feat: str, ratio=0.1, eps=0.01, corr_func=pearson_matrix
):
    """
    计算数据缺失比例对特征相关系数的影响
    :param data: pd.DataFrame, 数据
    :param feat: str, 特征
    :param ratio: float, 要删除的数据比例
    :param eps: float, 阈值
    :param corr_func: function, 相关函数
    :return : float, 相关系数的差异
    """
    eps /= 2
    cols = data.columns.tolist()
    if feat not in cols:
        raise ValueError(f"Feature {feat} not in data")
    corr_1 = corr_func(data)  # 计算原始数据的相关系数矩阵
    sampled_data = data.sample(frac=1 - ratio, axis=0)  # 随机删除部分数据
    corr_2 = corr_func(sampled_data)  # 计算删除部分数据后的相关系数矩阵
    diff = corr_1[feat] - corr_2[feat]  # 计算特定特征的相关系数差异
    return (diff.abs() < eps).sum() / data.shape[1]

def dcc_features(
    data: pd.DataFrame, feats: List[str], ratio=0.1, eps=0.01, corr_func=pearson_matrix
):
    """
    计算数据缺失比例对特征相关系数的影响
    :param data: pd.DataFrame, 数据
    :param feats: List[str], 特征
    :param ratio: float, 要删除的数据比例
    :param eps: float, 阈值
    :param corr_func: function, 相关函数
    :return : dict, 相关系数的差异
    """
    eps /= 2
    cols = data.columns.tolist()
    for feat in feats:
        if feat not in cols:
            raise ValueError(f"Feature {feat} not in data")
    corr_1 = corr_func(data)  # 计算原始数据的相关系数矩阵
    sampled_data = data.sample(frac=1 - ratio, axis=0)  # 随机删除部分数据
    corr_2 = corr_func(sampled_data)  # 计算删除部分数据后的相关系数矩阵
    results = {feat: 0 for feat in feats}
    for feat in feats:
        diff = corr_1[feat] - corr_2[feat]  # 计算特定特征的相关系数差异
        results[feat] = (diff.abs() < eps).sum() / data.shape[1]
    return results

def repeated_avg_dcc(
    data: pd.DataFrame, ratio=0.1, eps=0.01, corr_func=pearson_matrix, repeats=50
):
    results = []
    for _ in range(repeats):
        results.append(dcc(data, ratio, eps, corr_func))
    return sum(results) / repeats  # 计算多次重复后的平均值

def repeated_dcc(
    data: pd.DataFrame, ratio=0.1, eps=0.01, corr_func=pearson_matrix, repeats=50
):
    results = []
    for _ in range(repeats):
        results.append(dcc(data, ratio, eps, corr_func))
    return results  # 返回多次重复的结果

def repeated_avg_dcc_feature(
    data: pd.DataFrame,
    feat: str,
    ratio=0.1,
    eps=0.01,
    corr_func=pearson_matrix,
    repeats=10,
):
    results = []
    for _ in range(repeats):
        results.append(dcc_feature(data, feat, ratio, eps, corr_func))
    return sum(results) / repeats  # 计算多次重复后的平均值

def repeated_min_dcc(
    data: pd.DataFrame, ratio=0.1, eps=0.01, corr_func=pearson_matrix, repeats=100
):
    results = []
    for _ in range(repeats):
        results.append(dcc(data, ratio, eps, corr_func))
    return min(results)  # 计算多次重复后的最小值

def repeated_min_dcc_feature(
    data: pd.DataFrame,
    feat: str,
    ratio=0.1,
    eps=0.01,
    corr_func=pearson_matrix,
    repeats=100,
):
    results = []
    for _ in range(repeats):
        results.append(dcc_feature(data, feat, ratio, eps, corr_func))
    return min(results)  # 计算多次重复后的最小值

def redundancy(df: pd.DataFrame, corr_func=spearman_matrix):
    corr_df = corr_func(df)
    corr_df = corr_df.abs()
    m = df.shape[1]
    return (corr_df.sum().sum() - m) / (2 * m * (m - 1))  # 计算冗余度

def match_corr_method(method: str):
    """
    匹配相关系数计算方法
    :param method: str, 相关系数计算方法
    :return : 相关系数计算方法
    """
    match method:
        case "pearson":
            return pearson_matrix
        case "spearman":
            return spearman_matrix
        case "kendall":
            return kendall_matrix
        case "mutual_info":
            return mutual_info_matrix
        case _:
            return pearson_matrix

def dcc_diff(df1, df2, eps=0.04, corr_func=pearson_matrix):
    """
    计算两个数据框之间相关系数的差异
    :param df1: pd.DataFrame, 第一个数据框
    :param df2: pd.DataFrame, 第二个数据框
    :param eps: float, 阈值
    :param corr_func: function, 相关函数
    :return : float, 相关系数的差异
    """
    matrix1 = corr_func(df1)
    matrix2 = corr_func(df2)
    diff = matrix1 - matrix2
    return (float((diff.abs() < eps).sum().sum())) / (
        float(df1.shape[1]) * float(df1.shape[1])
    )

def dcc_diff_features(df1, df2, target_col, eps=0.04, corr_func=pearson_matrix):
    """
    计算两个数据框之间特征相关系数的差异
    :param df1: pd.DataFrame, 第一个数据框
    :param df2: pd.DataFrame, 第二个数据框
    :param target_col: str, 特征列名
    :param eps: float, 阈值
    :param corr_func: function, 相关函数
    :return : dict, 特征相关系数的差异
    """
    if target_col not in df1.columns or target_col not in df2.columns:
        raise ValueError(f"Feature {target_col} not in both dataframes")
    
    matrix1 = corr_func(df1)
    matrix2 = corr_func(df2)
    feats_cols = [col for col in df1.columns if col != target_col]
    feats2dcc = {}
    for col in feats_cols:
        diff = matrix1[col] - matrix2[col]
        feats2dcc[col] = float((diff.abs() < eps).sum()) / float(df1.shape[1])
    return feats2dcc


def dcc_difference(df1, df2, corr_func=pearson_matrix):
    """
    计算两个数据框之间相关系数的差异
    :param df1: pd.DataFrame, 第一个数据框
    :param df2: pd.DataFrame, 第二个数据框
    :param corr_func: function, 相关函数
    :return : float, 相关系数的差异
    """
    matrix1 = corr_func(df1)
    matrix2 = corr_func(df2)
    diff = matrix1 - matrix2
    return np.linalg.norm(diff, ord='fro')

def Kfold_dcc_diff(df, n_splits=10, eps=0.05, corr_func=pearson_matrix):
    """
    Perform K-Fold cross-validation on a DataFrame using a specified correlation function.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be split and evaluated.
    n_splits (int, optional): Number of folds for K-Fold cross-validation. Default is 10.
    eps (float, optional): A small value to avoid division by zero or other numerical issues. Default is 0.01.
    corr_func (function, optional): The correlation function to be used. Default is pearson_matrix.

    Returns:
    float: The average result of the dcc_diff function over all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    results = []
    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        results.append(dcc_diff(train_data, test_data, eps, corr_func))
    return sum(results) / n_splits


def Kfold_dcc_difference(df, n_splits=10, eps=0.05, corr_func=pearson_matrix):
    """
    Perform K-Fold cross-validation on a DataFrame using a specified correlation function.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be split and evaluated.
    n_splits (int, optional): Number of folds for K-Fold cross-validation. Default is 10.
    eps (float, optional): A small value to avoid division by zero or other numerical issues. Default is 0.01.
    corr_func (function, optional): The correlation function to be used. Default is pearson_matrix.

    Returns:
    float: The average result of the dcc_diff function over all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    results = []
    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        results.append(dcc_difference(train_data, test_data, corr_func))
    return sum(results) / n_splits