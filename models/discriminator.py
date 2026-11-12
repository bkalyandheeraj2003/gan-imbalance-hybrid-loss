import torch
import torch.nn as nn


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, class_embed_dim: int):
        """
        Conditional Discriminator / Critic for feature-vector data.
        """
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, class_embed_dim)
        in_dim = input_dim + class_embed_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embed = self.label_embedding(labels)
        features = torch.cat([x, embed], dim=1)
        return self.model(features)
