import pytest
import torch
import pytorch_lightning as pl
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from my_mlops_project.lightning_model import MyAwesomeModel
from my_mlops_project.lightning_train_test import ImageSamplesLogger, MetricsCallback, train


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/test_config.yaml")

@pytest.fixture
def mock_data():
    """Create mock training data."""
    train_data = torch.randn(100, 1, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    return torch.utils.data.TensorDataset(train_data, train_labels)

def test_model_creation(cfg):
    """Test that the model can be created with the given config."""
    model = MyAwesomeModel(cfg)
    assert isinstance(model, pl.LightningModule)
    
def test_train_val_split(cfg, mock_data):
    """Test that the train-validation split works correctly."""
    train_size = int(cfg.training.train_val_split * len(mock_data))
    val_size = len(mock_data) - train_size
    
    train_set, val_set = torch.utils.data.random_split(
        mock_data, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.training.seed)
    )
    
    assert len(train_set) == train_size
    assert len(val_set) == val_size
    assert len(train_set) + len(val_set) == len(mock_data)

@pytest.mark.parametrize("num_samples", [1, 5, 10])
def test_image_samples_logger(num_samples):
    """Test that the ImageSamplesLogger initializes correctly with different sample sizes."""
    logger = ImageSamplesLogger(num_samples=num_samples)
    assert logger.num_samples == num_samples

def test_metrics_callback():
    """Test that the MetricsCallback correctly stores metrics."""
    callback = MetricsCallback()
    
    # Create a mock trainer with callback metrics
    mock_trainer = MagicMock()
    mock_trainer.callback_metrics = {
        'train_loss': 0.5,
        'train_acc': 0.95
    }
    
    callback.on_train_epoch_end(mock_trainer, None)
    
    assert callback.final_metrics['train_loss'] == 0.5
    assert callback.final_metrics['train_acc'] == 0.95

def test_model_forward_pass(cfg):
    """Test that the model can perform a forward pass."""
    model = MyAwesomeModel(cfg)
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)  # Assuming MNIST dimensions
    
    output = model(x)
    assert output.shape == (batch_size, 10)  # Assuming 10 classes for MNIST

@pytest.mark.parametrize(
    "batch_size,num_workers", 
    [(32, 0), (64, 2), (128, 4)]
)
def test_dataloader_creation(mock_data, batch_size, num_workers):
    """Test that dataloaders can be created with different confogurations."""
    dataloader = torch.utils.data.DataLoader(
        mock_data,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    assert dataloader.batch_size == batch_size
    assert dataloader.num_workers == num_workers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_training_possible():
    """Test that GPU training is possible when CUDA is available."""
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    x = torch.randn(10, 1, 28, 28).to(device)
    assert x.is_cuda


