import os

import pytest

from bcclf import train


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory):
    """Train once into a temp directory so tests never touch real artifacts."""
    tmp = tmp_path_factory.mktemp("artifacts")
    os.environ["MODEL_PATH"] = str(tmp / "model.joblib")
    os.environ["REPORTS_DIR"] = str(tmp / "reports")
    metrics = train.run(models=["logistic_regression", "svm_rbf"])
    yield {"metrics": metrics, "dir": tmp}
