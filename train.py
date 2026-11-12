import torch
import torch.optim as optim
from models.generator import ConditionalGenerator
from models.discriminator import ConditionalDiscriminator
from losses.hybrid_loss import BalancedRepulsionLoss
from data.dataset import get_imbalanced_dataloader
import numpy as np

def main():
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset Initialization (Replace with your actual dataset arrays)
    num_samples = 1000
    feature_dim = 20
    num_classes = 2

    dummy_features = np.random.randn(num_samples, feature_dim).astype(np.float32)
    dummy_labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

    dataloader = get_imbalanced_dataloader(dummy_features, dummy_labels, batch_size=64)

    # 3. Model & Loss Initialization
    latent_dim = 10
    generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=num_classes, feature_dim=feature_dim).to(device)
    discriminator = ConditionalDiscriminator(feature_dim=feature_dim, num_classes=num_classes).to(device)

    criterion = BalancedRepulsionLoss(adv_loss_type="bce", repulsion_weight=0.01, mean_weight=2.0)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 4. Training Loop Epoch
    generator.train()
    discriminator.train()

    print("Starting training epoch...")
    for batch_idx, (real_data, labels) in enumerate(dataloader):
        real_data = real_data.to(device)
        labels = labels.to(device)
        batch_size = real_data.size(0)

        # Train Discriminator
        d_optimizer.zero_grad()
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_data = generator(noise, labels)

        real_pred = discriminator(real_data, labels)
        fake_pred = discriminator(fake_data.detach(), labels)
        
        d_loss = criterion.discriminator_loss(real_pred, fake_pred)
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_pred_g = discriminator(fake_data, labels)
        
        # Hybrid Loss integrating adversarial loss, mean-matching, and exponential repulsion
        g_loss = criterion.generator_loss(fake_pred_g, real_data=real_data, synthetic_data=fake_data)
        g_loss.backward()
        g_optimizer.step()

        if batch_idx == 0:
            print(f"Batch 0 Complete | D_Loss: {d_loss.item():.4f} | G_Loss: {g_loss.item():.4f}")
            break

    print("Training loop validation successful.")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
main()
