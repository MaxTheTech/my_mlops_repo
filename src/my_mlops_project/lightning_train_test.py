import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from my_mlops_project.data import corrupt_mnist
from my_mlops_project.lightning_model import MyAwesomeModel


# Define a custom callback to log input images to WandB
class ImageSamplesLogger(pl.Callback):
    """Custom callback to log input image samples to WandB during training."""

    def __init__(self, num_samples=5):
        super().__init__()
        self.num_samples = num_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log sample images every 100 batches."""
        if batch_idx % 100 == 0:
            images, _ = batch
            # Select first few images and log them to WandB
            samples = images[: self.num_samples]
            trainer.logger.experiment.log({"images": wandb.Image(samples.cpu(), caption="Input images")})


class MetricsCallback(pl.Callback):
    """Custom callback to track and store the final metrics."""

    def __init__(self):
        super().__init__()
        self.final_metrics = {}

    def on_train_epoch_end(self, trainer, pl_module):
        """Store the latest metrics at the end of each epoch."""
        self.final_metrics = {
            "train_loss": trainer.callback_metrics.get("train_loss", None),
            "train_acc": trainer.callback_metrics.get("train_acc", None),
        }


@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
def train(cfg) -> None:
    """Train a model on MNIST using PyTorch Lightning."""
    print("Training day and night")

    # Set random seeds for reproducibility
    pl.seed_everything(cfg.training.seed)

    # Print hyperparameters
    print(f"Learning rate: {cfg.optimizer.lr}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.epochs}")

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="corrupt_mnist",
        job_type="train",
        tags=["training"],
        config={"lr": cfg.optimizer.lr, "batch_size": cfg.training.batch_size, "epochs": cfg.training.epochs},
    )

    # Create model
    model = MyAwesomeModel(cfg)

    # Prepare data
    train_val_set, test_set = corrupt_mnist()

    # Split train_val_set into train and validation sets
    train_size = int(cfg.training.train_val_split * len(train_val_set))
    val_size = len(train_val_set) - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_val_set, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.training.seed)
    )

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.training.batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=cfg.training.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.training.batch_size)

    # Create metrics callback instance
    metrics_callback = MetricsCallback()

    # Define callbacks
    callbacks = [
        # Save best models based on validation loss
        # ModelCheckpoint(
        #     dirpath="models",
        #    filename='{epoch}-{val_loss:.2f}',
        #     monitor='val_loss',
        #     mode='min',
        #     save_top_k=3
        # ),
        EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min"),
        # Monitor learning rate changes
        LearningRateMonitor(logging_interval="step"),
        # Monitor device statistics (GPU usage, etc.)
        DeviceStatsMonitor(),
        # Log input image samples
        ImageSamplesLogger(num_samples=5),
        metrics_callback,  # Add our custom metrics callback
    ]

    # Initialize trainer with all the bells and whistles
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        # Automatically find best learning device (GPU/CPU)
        accelerator="mps" if torch.backends.mps.is_available() else "auto",
        devices=1,
        # Log gradients and model topology
        enable_model_summary=True,
        enable_checkpointing=True,
        # Progress bar updates
        enable_progress_bar=True,
        log_every_n_steps=10,
        # Detect gradient anomalies
        detect_anomaly=True,
        limit_train_batches=0.2,
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # After training, get the best model path and create a WandB artifact
    best_model_path = trainer.checkpoint_callback.best_model_path

    # Get final metrics from our callback
    final_metrics = metrics_callback.final_metrics

    # Create and log model artifact
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={
            # "best_val_loss": trainer.checkpoint_callback.best_model_score.item(),
            "final_train_loss": final_metrics.get("train_loss", None),
            "final_train_acc": final_metrics.get("train_acc", None),
        },
    )
    artifact.add_file(best_model_path)
    wandb_logger.experiment.log_artifact(artifact)

    print("Training complete!")

    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    train()
