import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

import wandb
from my_mlops_project.data import corrupt_mnist
from my_mlops_project.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg) -> None:
    """Train a model on MNIST."""
    print("Training day and night")

    # Set random seeds for reproducibility
    torch.manual_seed(cfg.training.seed)

    # Print hyperparameters
    print(f"Learning rate: {cfg.optimizer.lr}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.epochs}")

    wandb.init(
        project="corrupt_mnist",
        job_type="train",
        tags=["training"],
        config={"lr": cfg.optimizer.lr, "batch_size": cfg.training.batch_size, "epochs": cfg.training.epochs},
    )

    model = MyAwesomeModel(cfg).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.training.batch_size)

    loss_fn = getattr(torch.nn, cfg.loss.type)()
    optimizer = getattr(torch.optim, cfg.optimizer.type)(model.parameters(), lr=cfg.optimizer.lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.cpu().detach())})

        # Create a figure for ROC curves at the end of each epoch
        plt.figure(figsize=(10, 8))
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        # Use a color map for distinct colors for each class
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            RocCurveDisplay.from_predictions(
                one_hot,
                preds[:, class_id],
                name=f"Class {class_id}",
                plot_chance_level=(class_id == 2),
                color=colors[class_id],
            )

        # Enhance the plot with proper formatting
        plt.title(f"ROC Curves for All Classes - Epoch {epoch}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Correctly log the plot to wandb
        wandb.log({"roc_curves": wandb.Image(plt), "epoch": epoch})

        # Clean up to prevent memory leaks
        plt.close()

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)

    """
    print("Training complete")
    torch.save(model.state_dict(), cfg.training.model_checkpoint_path)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(cfg.training.training_statistics_path)
    """


if __name__ == "__main__":
    train()
