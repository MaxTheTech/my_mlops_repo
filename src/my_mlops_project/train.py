import matplotlib.pyplot as plt
import torch
import hydra

from my_mlops_project.model import MyAwesomeModel
from my_mlops_project.data import corrupt_mnist


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path="../../configs", config_name="config")
def train(cfg) -> None:
    """Train a model on MNIST."""
    print("Training day and night")

    # Set random seeds for reproducibility
    torch.manual_seed(cfg.training.seed)
    
    # Print hyperparameters
    print(f"Learning rate: {cfg.optimizer.lr}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.epochs}")

    model = MyAwesomeModel(cfg).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.training.batch_size)

    loss_fn = getattr(torch.nn, cfg.loss.type)()
    optimizer = getattr(torch.optim, cfg.optimizer.type)(model.parameters(), lr=cfg.optimizer.lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.training.epochs):
        model.train()
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

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), cfg.training.model_checkpoint_path)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(cfg.training.training_statistics_path)


if __name__ == "__main__":
    train()
