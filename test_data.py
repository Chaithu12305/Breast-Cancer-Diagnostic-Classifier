import numpy as np
import pytest

from bcclf.data import load_dataset, split_data, validate_features


def test_dataset_shape():
    X, y, names = load_dataset()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert names == ["malignant", "benign"]


def test_split_is_stratified():
    X, y, _ = load_dataset()
    _, _, y_tr, y_te = split_data(X, y)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert len(y_tr) + len(y_te) == len(y)


def test_validation_rejects_nan():
    X, _, _ = load_dataset()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_features(X)
