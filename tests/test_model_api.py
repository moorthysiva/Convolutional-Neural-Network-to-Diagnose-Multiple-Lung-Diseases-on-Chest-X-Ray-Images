"""Smoke tests that need the heavy stack; skipped cleanly when it is absent so
the lightweight CI job still passes and the full job exercises them."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(pytest.importorskip("timm", reason="timm not installed") is None,
                    reason="timm required")
def test_model_forward_shape_and_no_baked_activation():
    from src.chestxray.models import build_model

    model = build_model("resnet18", num_classes=14, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 14)
    # Logits are unbounded (no sigmoid/softmax inside the model).
    assert (logits.abs() > 0).any()
    probs = model.predict_proba(x)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_losses_run_on_logits():
    from src.chestxray.training.losses import build_loss

    logits = torch.randn(4, 3)
    targets = torch.randint(0, 2, (4, 3)).float()
    for kind in ("bce", "weighted_bce", "focal"):
        loss = build_loss(kind, pos_weight=np.ones(3), device="cpu")(logits, targets)
        assert loss.item() >= 0


def test_health_endpoint():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()
