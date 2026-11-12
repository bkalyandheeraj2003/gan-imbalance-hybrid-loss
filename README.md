# Hybrid Loss GAN for Class-Imbalanced Synthetic Data

Experimental PyTorch repository evaluating a hybrid loss function designed to prevent mode collapse in minority classes during synthetic data generation.

## Kaggle Quickstart
```bash
!git clone [https://github.com/bkalyandheeraj2003/gan-imbalance-hybrid-loss.git](https://github.com/bkalyandheeraj2003/gan-imbalance-hybrid-loss.git)
%cd gan-imbalance-hybrid-loss
cat << 'EOF' > requirements.txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pyyaml>=6.0
umap-learn>=0.5.3
matplotlib>=3.7.0
seaborn>=0.12.0
