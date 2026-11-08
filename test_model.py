import pytest

from bcclf.data import load_dataset
from bcclf.model import build_pipeline
from bcclf.predict import load_artifact, predict_records


def test_unknown_model():
    with pytest.raises(ValueError):
        build_pipeline("nope")


def test_training_quality(trained_env):
    m = trained_env["metrics"]
    assert m["accuracy"] > 0.93
    assert m["roc_auc"] > 0.97
    assert m["recall_malignant"] > 0.90


def test_artifacts_written(trained_env):
    d = trained_env["dir"]
    assert (d / "model.joblib").exists()
    for f in ["metrics.json", "model_comparison.csv", "confusion_matrix.png", "roc_curve.png"]:
        assert (d / "reports" / f).exists()


def test_predict_and_missing_feature(trained_env):
    art = load_artifact()
    X, _, _ = load_dataset()
    rec = X.iloc[0].to_dict()
    out = predict_records(art, [rec])[0]
    assert out["prediction"] in {"malignant", "benign"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    rec.pop(next(iter(rec)))
    with pytest.raises(ValueError):
        predict_records(art, [rec])
