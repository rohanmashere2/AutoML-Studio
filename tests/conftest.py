"""Shared test fixtures for AutoML Studio test suite."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def classification_df():
    """Standard binary classification dataset (200 samples, 10 features)."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


@pytest.fixture
def regression_df():
    """Standard regression dataset (200 samples, 10 features)."""
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


@pytest.fixture
def mixed_df():
    """Dataset with mixed types, missing values, and outliers."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n).astype(float),
        'salary': np.random.normal(50000, 15000, n),
        'gender': np.random.choice(['M', 'F'], n),
        'city': np.random.choice(
            ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix',
             'Dallas', 'Miami', 'Denver', 'Seattle', 'Boston',
             'Atlanta', 'Portland'],
            n,
        ),
        'score': np.random.uniform(0, 100, n),
        'target': np.random.choice([0, 1], n),
    })
    # Add missing values
    df.loc[np.random.choice(n, 15, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(n, 10, replace=False), 'salary'] = np.nan
    df.loc[np.random.choice(n, 8, replace=False), 'city'] = np.nan
    # Add outliers
    df.loc[0, 'salary'] = 500000
    df.loc[1, 'salary'] = -10000
    return df


@pytest.fixture
def empty_df():
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def single_row_df():
    """Single-row DataFrame."""
    return pd.DataFrame({'a': [1.0], 'b': [2.0], 'target': [0]})


@pytest.fixture
def all_nan_df():
    """DataFrame with all NaN numeric values."""
    return pd.DataFrame({
        'a': [np.nan] * 10,
        'b': [np.nan] * 10,
        'target': [0, 1] * 5,
    })


@pytest.fixture
def classification_profile():
    """Minimal profile dict for classification."""
    return {
        'target_column': 'target',
        'problem_type': 'classification',
        'n_cols': 11,
        'n_rows': 200,
        'numeric_columns': [f'feature_{i}' for i in range(10)],
        'categorical_columns': [],
        'text_columns': [],
    }


@pytest.fixture
def regression_profile():
    """Minimal profile dict for regression."""
    return {
        'target_column': 'target',
        'problem_type': 'regression',
        'n_cols': 11,
        'n_rows': 200,
        'numeric_columns': [f'feature_{i}' for i in range(10)],
        'categorical_columns': [],
        'text_columns': [],
    }


@pytest.fixture
def tmp_csv(classification_df, tmp_path):
    """Save classification_df to a temp CSV and return the path."""
    path = tmp_path / "test_data.csv"
    classification_df.to_csv(path, index=False)
    return str(path)
