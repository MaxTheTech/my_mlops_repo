import torch
from torch import nn
import hydra
from omegaconf import DictConfig

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        
        layers = []
        in_channels = cfg.model.input_channels
        
        # Build conv layers dynamically from config
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
            x = torch.max_pool2d(x, self.pool_size, self.pool_stride)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    model = MyAwesomeModel(cfg)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, cfg.model.input_channels, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()