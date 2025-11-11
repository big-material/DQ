import pandas as pd
import numpy as np
from typing import Callable
from scipy.stats import chatterjeexi
from joblib import Parallel, delayed
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from scipy.spatial.distance import minkowski

def calculate_pair(df: pd.DataFrame, method: Callable, i: int, j: int) -> tuple:
    value = method(df.iloc[:, i], df.iloc[:, j])
    return (i, j, value)

def corr(df: pd.DataFrame, method: Callable) -> pd.DataFrame:
    """
    Calculate the correlation matrix of a given dataframe using joblib for parallel processing
    """
    cols = len(df.columns)
    corr_matrix = np.zeros((cols, cols))
    
    results = Parallel(n_jobs=-1)(delayed(calculate_pair)(df, method, i, j) 
                                  for i in range(cols) for j in range(i, cols))
    
    for i, j, value in results:
        corr_matrix[i, j] = value
        corr_matrix[j, i] = value
            
    return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)


def pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the pearson correlation matrix of a given dataframe
    """
    return df.corr(method='pearson')


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the spearman correlation
    """
    return df.corr(method='spearman')


def kendall_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the kendall correlation
    """
    return df.corr(method='kendall')


def joint_entropies(data: np.ndarray, nbins) -> np.ndarray:
    """
    Calculate the joint entropies of a given dataset
    """
    n_variables = data.shape[-1]
    n_samples = data.shape[0]
    if nbins == None:
        nbins = int((n_samples/5)**.5)
    histograms2d = np.zeros((n_variables, n_variables, nbins, nbins))
    for i in range(n_variables):
        for j in range(n_variables):
            histograms2d[i, j] = np.histogram2d(
                data[:, i], data[:, j], bins=nbins)[0]
    probs = histograms2d / len(data) + 1e-100
    joint_entropies = -(probs * np.log2(probs)).sum((2, 3))
    return joint_entropies


def mutual_info_matrix(df: pd.DataFrame, nbins=None, normalized=True):
    """
    Calculates the mutual information matrix for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - nbins (int, optional): The number of bins to use for discretization. If not provided, the default number of bins will be used.
    - normalized (bool, optional): Whether to normalize the mutual information matrix. Default is True.

    Returns:
    - pd.DataFrame: The mutual information matrix as a DataFrame, with column and row names corresponding to the input DataFrame's columns.

    """
    data = df.to_numpy()
    n_variables = data.shape[-1]
    j_entropies = joint_entropies(data, nbins)
    entropies = j_entropies.diagonal()
    entropies_tile = np.tile(entropies, (n_variables, 1))
    sum_entropies = entropies_tile + entropies_tile.T
    mi_matrix = sum_entropies - j_entropies
    if normalized:
        mi_matrix = mi_matrix * 2 / sum_entropies
    return pd.DataFrame(mi_matrix, index=df.columns, columns=df.columns)

def partial_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the partial correlation matrix of a given dataframe
    """
    return df.pcorr()

def target_mutual_info(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Calculate the mutual information score of a given dataframe
    """
    mi_df = mutual_info_matrix(df)
    return mi_df[target]

# def xi_matrix(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculate the xi correlation matrix of a given dataframe
#     """
#     def xi_corr(a, b):
#         xi, _ = xicorr(a, b)
#         return xi
#     return corr(df, xi_corr)



def to_distribution(lst):
    arr = np.array(lst, dtype=np.float64)
    kde = gaussian_kde(arr)
    x = np.linspace(np.min(arr), np.max(arr), 1000)
    pdf = kde(x)
    pdf /= np.sum(pdf)  # 归一化为概率分布
    return pdf

def compute_kl_divergence(p, q):
    """计算 KL 散度 D_KL(P || Q)"""
    P = to_distribution(p)
    Q = to_distribution(q)
    return np.sum(rel_entr(P, Q))

def compute_js_divergence(p, q):
    """计算 JS 散度 D_JS(P || Q)"""
    P = to_distribution(p)
    Q = to_distribution(q)
    js_distance = jensenshannon(P, Q)
    return js_distance


def zscore(x):
    return (np.array(x) - np.mean(x)) / np.std(x)

def compute_wd_distance(p, q):
    """计算 Wasserstein 距离"""
    from scipy.stats import wasserstein_distance
    p = zscore(p)
    q = zscore(q)
    return wasserstein_distance(p, q)


def js_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Jensen-Shannon correlation matrix of a given dataframe
    """
    
    return corr(df, compute_js_divergence)

def kl_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the KL divergence matrix of a given dataframe
    """
    return corr(df, compute_kl_divergence)

def wd_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Wasserstein distance matrix of a given dataframe
    """
    return corr(df, compute_wd_distance)


def minkwoski_dis(A, B, p=2):
    """Calculate Minkowski distance"""
    A = (A - A.min()) / (A.max() - A.min())
    B = (B - B.min()) / (B.max() - B.min())
    return minkowski(A, B, p)

def Chebyshev_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Chebyshev distance matrix of a given dataframe
    """
    return corr(df, lambda a, b: minkowski(a, b, p=np.inf))

def manhattan_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Manhattan distance matrix of a given dataframe
    """
    return corr(df, lambda a, b: minkowski(a, b, p=1))

def euclidean_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Euclidean distance matrix of a given dataframe
    """
    return corr(df, lambda a, b: minkowski(a, b, p=2))

def rbf_kernel(x, sigma=None):
    x = np.asarray(x).reshape(-1, 1)
    # 计算 pairwise squared Euclidean distance: ||x_i - x_j||^2
    dists = (x - x.T) ** 2  # 结果是 shape (n, n)
    if sigma is None:
        sigma = np.median(np.sqrt(dists))
        if sigma == 0:
            sigma = 1e-6
    K = np.exp(-dists / (2 * sigma ** 2))
    return K

def center_kernel(K):
    """
    Center a kernel matrix using H = I - 1/n * 11ᵀ
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def hsic(X, Y, sigma_x=None, sigma_y=None):
    """
    Compute HSIC between two 1D arrays X and Y.
    """
    K = rbf_kernel(X, sigma_x)
    L = rbf_kernel(Y, sigma_y)
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    hsic_val = np.trace(Kc @ Lc) / ((len(X) - 1) ** 2)
    return hsic_val

def nhsic(X, Y, sigma_x=None, sigma_y=None):
    """
    Compute normalized HSIC between two 1D arrays.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    hsic_xy = hsic(X, Y, sigma_x, sigma_y)
    hsic_xx = hsic(X, X, sigma_x, sigma_x)
    hsic_yy = hsic(Y, Y, sigma_y, sigma_y)
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

def nhsic_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the normalized HSIC matrix of a given dataframe.
    """
    return corr(df, nhsic)

def xi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the xi correlation matrix of a given dataframe.
    """
    def xi_corr(a, b):
        res = chatterjeexi(a, b)
        return res.statistic
    return corr(df, xi_corr)

def dcor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance correlation matrix of a given dataframe.
    """
    import dcor
    return corr(df, dcor.distance_correlation)