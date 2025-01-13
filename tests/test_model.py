import torch
from omegaconf import OmegaConf
import pytest

from my_mlops_project.model import MyAwesomeModel

@pytest.fixture
def cfg():
    return OmegaConf.load("configs/test_config.yaml")
    
def test_config(cfg):
    assert cfg["optimizer"]["lr"] == 0.001
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["epochs"] == 2
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["train_val_split"] == 0.8

def test_model(cfg):
    model = MyAwesomeModel(cfg)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
