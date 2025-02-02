import hydra
import torch


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
def preprocess_data(cfg) -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"data/raw/train_images_{i}.pt"), weights_only=True)
        train_target.append(torch.load(f"data/raw/train_target_{i}.pt"), weights_only=True)
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"data/raw/test_images.pt", weights_only=True)
    test_target: torch.Tensor = torch.load(f"data/raw/test_target.pt", weights_only=True)

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    if cfg.preprocessing.normalize:
        train_images = normalize(train_images)
        test_images = normalize(test_images)
        print("Data normalized")
    else:
        print("Data not normalized")

    torch.save(train_images, f"data/processed/train_images.pt")
    torch.save(train_target, f"data/processed/train_target.pt")
    torch.save(test_images, f"data/processed/test_images.pt")
    torch.save(test_target, f"data/processed/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt", weights_only=True)
    train_target = torch.load("data/processed/train_target.pt", weights_only=True)
    test_images = torch.load("data/processed/test_images.pt", weights_only=True)
    test_target = torch.load("data/processed/test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    preprocess_data()
