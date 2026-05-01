# tests/test_preprocessor.py


def test_customerid_dropped(preprocessed_df):
    """Test that customerid column is dropped after preprocessing."""
    assert "customerid" not in preprocessed_df.columns


def test_totalcharges_numeric(preprocessed_df):
    """Test that totalcharges is converted to numeric."""
    assert preprocessed_df["totalcharges"].dtype in ["float64", "float32"]


def test_encode_columns_binary(preprocessed_df):
    """Test that binary columns are encoded as 0/1."""
    assert preprocessed_df["gender"].isin([0, 1]).all()
    assert preprocessed_df["partner"].isin([0, 1]).all()
    assert preprocessed_df["dependents"].isin([0, 1]).all()


def test_no_duplicates(preprocessed_df):
    """Test that duplicates are removed after preprocessing."""
    assert preprocessed_df.duplicated().sum() == 0


def test_shape(raw_df, preprocessed_df):
    """Test that preprocessing does not add rows."""
    assert preprocessed_df.shape[0] <= raw_df.shape[0]


def test_no_high_missing_values(missing_percentage):
    """Test that no column has more than 20% missing values."""
    high_missing = missing_percentage[missing_percentage > 20]
    assert len(high_missing) == 0, (
        f"Columns with >20% missing values: {high_missing.to_dict()}"
    )