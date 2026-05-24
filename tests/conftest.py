"""
Shared pytest fixtures for AutoML Studio tests.
"""

import os
import pytest
import pandas as pd


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def small_dataset():
    """Load the 50-row test fixture CSV."""
    path = os.path.join(FIXTURES_DIR, 'small_dataset.csv')
    return pd.read_csv(path)


@pytest.fixture
def small_dataset_path():
    """Return the path to the small fixture CSV."""
    return os.path.join(FIXTURES_DIR, 'small_dataset.csv')
