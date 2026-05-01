# tests/conftest.py

import pytest
from src.data.loader import loader_pipeline
from src.data.preprocess import preprocess_pipeline


@pytest.fixture
def raw_df():
    """Load 10 rows of real raw data."""
    df = loader_pipeline()
    return df.head(10)


@pytest.fixture
def preprocessed_df(raw_df):
    """Preprocess 10 rows of real raw data."""
    return preprocess_pipeline(raw_df.copy())


@pytest.fixture
def missing_percentage(raw_df):
    """Calculate missing percentage for each column."""
    return (raw_df.isnull().sum() / len(raw_df)) * 100