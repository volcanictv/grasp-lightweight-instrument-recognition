from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from surgical_ai.training import trainer


class RecordingLoss(nn.CrossEntropyLoss):
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def run_fit(tmp_path: Path, monkeypatch, select_best: bool) -> list[int]:
    saved_at: list[int] = []
    loss_fn = RecordingLoss()
    monkeypatch.setattr(trainer.torch, "save", lambda _state, _path: saved_at.append(loss_fn.epoch))
    val_f1 = iter([0.9, 0.5, 0.4, 0.3])  # validation gets worse every epoch, so the best epoch is the first

    def fake_evaluate(_model, loader, _loss, _names, _device):
        # called twice per epoch: train metrics, then validation metrics
        return 0.0, SimpleNamespace(macro_f1=0.0 if loader.dataset is train_loader.dataset else next(val_f1))

    model = nn.Linear(4, 2)
    data = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    train_loader = DataLoader(data, batch_size=4)
    val_loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=4)
    trainer.fit(model, train_loader, val_loader, loss_fn, torch.optim.SGD(model.parameters(), lr=0.1), ["a", "b"],
                torch.device("cpu"), epochs=4, checkpoint_path=tmp_path / "best.pt", evaluate_fn=fake_evaluate, select_best=select_best)
    return saved_at


def test_default_keeps_the_best_validation_epoch(tmp_path, monkeypatch):
    assert run_fit(tmp_path, monkeypatch, select_best=True) == [1]


def test_fixed_schedule_keeps_only_the_last_epoch(tmp_path, monkeypatch):
    assert run_fit(tmp_path, monkeypatch, select_best=False) == [4]
