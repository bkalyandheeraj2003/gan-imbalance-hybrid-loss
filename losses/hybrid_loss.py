import torch
import torch.nn as nn


class BalancedRepulsionLoss(nn.Module):
    def __init__(self, adv_loss_type: str = "bce", repulsion_weight: float = 0.01, mean_weight: float = 2.0):
        """
        Balanced Repulsion Loss combining adversarial objective, distribution mean-matching, 
        and an exponential repulsion penalty to actively prevent mode collapse in imbalanced GANs.
        """
        super().__init__()
        self.adv_loss_type = adv_loss_type
        self.rep_w = repulsion_weight
        self.mean_w = mean_weight

        if adv_loss_type == "bce":
            self.bce = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        if self.adv_loss_type == "bce":
            real_targets = torch.ones_like(real_pred)
            fake_targets = torch.zeros_like(fake_pred)
            loss_real = self.bce(real_pred, real_targets)
            loss_fake = self.bce(fake_pred, fake_targets)
            return (loss_real + loss_fake) * 0.5
        elif self.adv_loss_type == "wasserstein":
            return torch.mean(fake_pred) - torch.mean(real_pred)

    def generator_loss(
        self,
        fake_pred: torch.Tensor,
        real_data: torch.Tensor = None,
        synthetic_data: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Standard GAN Adversarial Loss
        if self.adv_loss_type == "bce":
            targets = torch.ones_like(fake_pred)
            g_loss = self.bce(fake_pred, targets)
        elif self.adv_loss_type == "wasserstein":
            g_loss = -torch.mean(fake_pred)

        # If feature tensors are provided, compute hybrid penalties
        if real_data is not None and synthetic_data is not None:
            # 2. Mean Matching (Centers the synthetic distribution toward the real distribution)
            mean_real = real_data.mean(dim=0)
            mean_fake = synthetic_data.mean(dim=0)
            mean_loss = torch.mean((mean_real - mean_fake) ** 2)

            # 3. Repulsion Penalty (Pushes generated points apart to prevent mode collapse/clustering)
            dist = torch.cdist(synthetic_data, synthetic_data, p=2)
            repulsion = torch.exp(-dist).mean()

            # Combined Hybrid Loss
            total_loss = g_loss + (self.mean_w * mean_loss) + (self.rep_w * repulsion)
            return total_loss

        return g_loss
