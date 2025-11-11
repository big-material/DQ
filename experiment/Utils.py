import pandas as pd
import numpy as np
import random
from sklearn.datasets import (
    make_regression,
    make_sparse_uncorrelated,
    make_classification,
)
from numpy import ndarray


def generate_random_df(n: int, m: int) -> pd.DataFrame:
    """
    Generate a random dataset with specified parameters.

    Parameters:
        n (int): Number of samples.
        m (int): Number of features.

    Returns:
        pd.DataFrame: Generated random dataset with 'n' samples and 'm' features.
    """
    return pd.DataFrame(
        np.random.rand(n, m + 1),
        columns=[f"feature_{i}" for i in range(m)] + ["target"],
    )


def make_unlinear_data(
    n: int, m: int, n_informative: int = 10, noise=0.1
) -> tuple[ndarray, ndarray]:
    """
    Generate a unlinear dataset with specified parameters.

    Parameters:
        n (int): Number of samples.
        m (int): Number of features.
        n_informative (int, optional): Number of informative features. Defaults to 10.
        noise (float, optional): Standard deviation of the Gaussian noise applied to the output. Defaults to 0.1.

    Returns:
        pd.DataFrame: Generated unlinear dataset with 'n' samples and 'm' features.
    """
    X = np.random.rand(n, m)
    informative_indices = random.sample(range(m), n_informative)
    y = np.zeros(n)
    nonlinear_transformation = random.choice([np.sin, np.cos, np.exp])
    for idx in informative_indices:
        y += nonlinear_transformation(X[:, idx])  # Apply nonlinear transformation
    y += noise * np.random.randn(n)
    return X, y


def generate_regression_df(
    n: int, m: int, n_informative: int = 10, noise=0.1
) -> pd.DataFrame:
    """
    Generate a regression dataset with specified parameters.

    Parameters:
        n (int): Number of samples.
        m (int): Number of features.
        n_informative (int, optional): Number of informative features. Defaults to 10.
        noise (float, optional): Standard deviation of the Gaussian noise applied to the output. Defaults to 0.1.

    Returns:
        pd.DataFrame: Generated regression dataset with 'n' samples and 'm' features.
    """
    X, y = make_regression(
        n_samples=n, n_features=m, n_informative=n_informative, noise=noise
    )
    columns = [f"feature_{i}" for i in range(m)]
    columns.append("target")
    data = np.column_stack((X, y))
    return pd.DataFrame(data, columns=columns)


def mixed_random_regression_data(
    n: int, m: int, ratio: float = 0.2, n_informative: int = 10, noise=0.1
) -> pd.DataFrame:
    """
    Generate a mixed regression dataset with specified parameters.

    Parameters:
        n (int): Number of samples.
        m (int): Number of features.
        ratio (float, optional): Ratio of linear data. Defaults to 0.2.
        n_informative (int, optional): Number of informative features. Defaults to 10.
        noise (float, optional): Standard deviation of the Gaussian noise applied to the output. Defaults to 0.1.

    Returns:
        pd.DataFrame: Generated regression dataset with 'n' samples and 'm' features.
    """
    if ratio == 0.0:
        return generate_random_df(n, m)
    elif ratio == 1.0:
        return generate_regression_df(n, m, n_informative, noise)
    else:
        n1 = int(n * ratio)
        if n1 == 0:
            return generate_random_df(n, m)
        n2 = n - n1
        data1 = generate_regression_df(n1, m, n_informative, noise)
        data2 = generate_random_df(n2, m)
        return pd.concat([data1, data2], axis=0)


def mix(ratio, linear_data, random_data, seed=0):
    """
    Mix linear and random data.
    ratio: float, the ratio of linear data in the mixed data.
    linear_data: pd.DataFrame, linear data.
    random_data: pd.DataFrame, random data.
    return: pd.DataFrame, mixed data.
    """
    np.random.seed(int(seed * 1000))
    # Calculate how many samples to take from each dataset
    n_linear = int(ratio * len(linear_data))
    n_random = len(linear_data) - n_linear

    # Sample from each dataset
    part_linear_data = linear_data.sample(n=n_linear)
    part_random_data = random_data.sample(n=n_random)

    # Combine datasets
    mixed_data = pd.concat([part_linear_data, part_random_data])
    mixed_data.reset_index(drop=True, inplace=True)
    return mixed_data


def split_dataframe(df, n_parts=30):
    """
    Split a DataFrame into approximately equal parts and return an iterator.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to split
    n_parts : int, default=5
        Number of parts to split the DataFrame into

    Returns:
    --------
    iterator
        An iterator that yields each part of the DataFrame
    """
    # Calculate the approximate size of each part
    part_size = len(df) // n_parts
    remainder = len(df) % n_parts

    start = 0
    for i in range(n_parts):
        # Add one to the part size for the first 'remainder' parts to distribute the remainder
        current_part_size = part_size + (1 if i < remainder else 0)
        end = start + current_part_size

        # Yield the current part
        yield df.iloc[start:end]

        # Update start position for the next part
        start = end


def random_replace_rows(df_a, df_b, random_state=None):
    """
    Randomly replace rows in dataframe A with all rows from dataframe B.

    Parameters:
    -----------
    df_a : pandas.DataFrame
        The dataframe to have rows replaced
    df_b : pandas.DataFrame
        The dataframe providing replacement rows
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Modified dataframe A
    """
    # Make copies to avoid modifying the original dataframes
    df_a_copy = df_a.copy().astype(np.float64)
    df_b_copy = df_b.copy().astype(np.float64)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Ensure the dataframes have the same columns
    if not set(df_a_copy.columns).issubset(set(df_b_copy.columns)):
        raise ValueError("Dataframe B must contain all columns of dataframe A")

    # resort the columns of df_b to match df_a
    df_b_copy = df_b_copy[df_a_copy.columns]

    # Number of rows to replace (all rows from df_b)
    n_replace = len(df_b_copy)

    # If df_b has more rows than df_a, we can only replace up to the size of df_a
    if n_replace > len(df_a_copy):
        n_replace = len(df_a_copy)

    # Generate random indices from df_a to replace
    replace_indices = np.random.choice(len(df_a_copy), size=n_replace, replace=False)

    # Replace selected rows with rows from df_b
    df_a_copy.iloc[replace_indices] = df_b_copy.iloc[:n_replace].values

    return df_a_copy


def add_noise_to_dataframe(
    df: pd.DataFrame,
    noise_ratio: float,
    noise_scale: float = 0.1,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Adds Gaussian noise to a specified ratio of rows in a DataFrame.

    The noise added to each column is scaled by the column's standard deviation.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    noise_ratio : float
        The proportion of rows to which noise should be added (between 0 and 1).
    noise_scale : float, default=0.1
        The scaling factor for the noise, applied to the standard deviation of each column.
        Noise magnitude will be N(0, noise_scale * column_std_dev).
    random_state : int, optional
        Seed for the random number generator for reproducibility.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with noise added to the specified rows.
    """
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError("noise_ratio must be between 0 and 1.")

    df_noisy = df.copy()
    n_rows = len(df_noisy)
    n_noisy_rows = int(n_rows * noise_ratio)

    if n_noisy_rows == 0:
        return df_noisy  # No noise to add

    if random_state is not None:
        np.random.seed(random_state)

    # Select random row indices to add noise
    noisy_indices = np.random.choice(n_rows, size=n_noisy_rows, replace=False)

    # trans all col to float64
    df_noisy = df_noisy.astype(np.float64)

    for col in df_noisy.columns:
        col_std = df_noisy[col].std()
        # Avoid adding noise if std dev is zero or NaN
        if pd.isna(col_std) or col_std == 0:
            continue
        # Generate noise: N(0, noise_scale * col_std)
        noise = np.random.normal(
            loc=0.0, scale=noise_scale * col_std, size=n_noisy_rows
        )
        # Add noise to the selected rows for the current column
        df_noisy.loc[noisy_indices, col] += noise

    return df_noisy


def calculate_area_under_curve(Y, X):
    """
    Calculates the area under the curve defined by points (X, Y) using the trapezoidal rule.

    Assumes the points form a curve above the x-axis and connects to the y-axis.
    The area calculated is between the curve, the x-axis, and the y-axis.

    Parameters:
    -----------
    X : list or numpy.ndarray
        List of x-coordinates.
    Y : list or numpy.ndarray
        List of y-coordinates. Must be the same length as X.

    Returns:
    --------
    float
        The calculated area. Returns 0 if input lists are empty or have only one point.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length.")
    if len(X) < 2:
        return 0.0  # Cannot form an area with less than 2 points

    # Combine X and Y into pairs and sort by X coordinate
    points = sorted(zip(X, Y))
    X_sorted = [p[0] for p in points]
    Y_sorted = [p[1] for p in points]

    # Calculate area using the trapezoidal rule
    area = np.trapz(Y_sorted, X_sorted)

    # Ensure area is non-negative (useful if curve dips below x-axis, though the prompt implies positive area)
    return max(0.0, area)


def generate_random_according_df(n: int, m: int, df: pd.DataFrame):
    cols = df.columns
    new_data = []
    for i in range(n):
        new_row = []
        for j in range(m):
            new_row.append(random.uniform(df[cols[j]].min(), df[cols[j]].max()))
        new_data.append(new_row)
    new_df = pd.DataFrame(new_data, columns=cols)
    return new_df


def generate_linear_according_df(n: int, m: int, df: pd.DataFrame):
    X, y = make_regression(n_samples=n, n_features=m - 1, noise=0.1, random_state=42)[
        :2
    ]
    data = np.column_stack((X, y))
    df_linear = pd.DataFrame(data, columns=df.columns)
    # normalize the data to the range of df
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_linear[col] = (df_linear[col] - df_linear[col].min()) / (
            df_linear[col].max() - df_linear[col].min()
        ) * (max_val - min_val) + min_val
    return df_linear


def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Calculate the adjusted R-squared score.

    Parameters:
    -----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    n_features : int
        Number of features used in the model.

    Returns:
    --------
    float
        Adjusted R-squared score.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan  # Handle empty inputs gracefully

    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    n = len(y_true)

    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return adjusted_r2


def generate_classification_df(
    n: int, m: int, n_informative: int = 10, noise=0.1
) -> pd.DataFrame:
    """
    Generate a classification dataset with specified parameters.

    Parameters:
        n (int): Number of samples.
        m (int): Number of features.
        n_informative (int, optional): Number of informative features. Defaults to 10.
        noise (float, optional): Standard deviation of the Gaussian noise applied to the output. Defaults to 0.1.

    Returns:
        pd.DataFrame: Generated classification dataset with 'n' samples and 'm' features.
    """

    # n_classes(6) * n_clusters_per_class(2) must be smaller or equal 2**n_informative(2)=4
    n_classes = np.random.randint(2, min(10, n_informative))
    X, y = make_classification(
        n_samples=n,
        n_features=m,
        n_informative=n_informative,
        n_classes=n_classes,
        n_redundant=0,
        n_repeated=0,
    )
    columns = [f"feature_{i}" for i in range(m)]
    columns.append("target")
    data = np.column_stack((X, y))
    return pd.DataFrame(data, columns=columns)
