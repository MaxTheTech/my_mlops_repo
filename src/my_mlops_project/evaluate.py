import torch
import hydra

from my_mlops_project.model import MyAwesomeModel
from data import corrupt_mnist


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path="../../configs", config_name="config")
def evaluate(cfg) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(cfg.evaluate.model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(cfg.training.model_checkpoint_path))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.training.batch_size)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    evaluate()
