import torch
import hydra
from torch import nn
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

class MyAwesomeModel(pl.LightningModule):
    """
    PyTorch Lightning version of MyAwesomeModel.
    Adds training, validation, and test loops plus optimization configuration.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters(cfg)
        
        # Build convolutional layers
        layers = []
        in_channels = cfg.model.input_channels
        
        for conv in cfg.model.conv_layers:
            layers.append(
                nn.Conv2d(
                    in_channels, 
                    conv.filters, 
                    conv.kernel_size, 
                    conv.stride
                )
            )
            in_channels = conv.filters
            
        self.conv_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(cfg.model.dropout_rate)
        self.fc1 = nn.Linear(cfg.model.conv_layers[-1].filters, cfg.model.output_dim)
        
        # Store pooling parameters
        self.pool_size = cfg.model.pool_size
        self.pool_stride = cfg.model.pool_stride

        # Initialize metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.output_dim)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.output_dim)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.output_dim)

        # Store training parameters
        self.lr = cfg.optimizer.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
            x = torch.max_pool2d(x, self.pool_size, self.pool_stride)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
                
        """
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
        """

    def training_step(self, batch, batch_idx):
        """Training step logic."""
        img, target = batch
        y_pred = self(img)
        loss = nn.functional.cross_entropy(y_pred, target)
        
        # Calculate and log accuracy
        acc = self.train_accuracy(y_pred, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step logic."""
        img, target = batch
        y_pred = self(img)
        loss = nn.functional.cross_entropy(y_pred, target)
        
        # Calculate and log accuracy
        acc = self.train_accuracy(y_pred, target)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step logic."""
        img, target = batch
        y_pred = self(img)
        loss = nn.functional.cross_entropy(y_pred, target)
        
        # Calculate and log accuracy
        acc = self.train_accuracy(y_pred, target)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    # Create model
    model = MyAwesomeModel(cfg)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',  # Automatically detect GPU/CPU
        devices=1,
        log_every_n_steps=10,
    )
    
    # Print model summary
    print(f"Model architecture:")
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Optional: Test with dummy data
    dummy_input = torch.randn(1, cfg.model.input_channels, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()