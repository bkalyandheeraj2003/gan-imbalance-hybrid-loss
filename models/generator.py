import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, class_embed_dim: int, output_dim: int):
        """
        Conditional Generator for tabular or feature-vector data.
        """
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, class_embed_dim)
        input_dim = latent_dim + class_embed_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embed = self.label_embedding(labels)
        x = torch.cat([z, embed], dim=1)
        return self.model(x)
