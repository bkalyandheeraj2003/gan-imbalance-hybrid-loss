import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture

def compute_mode_coverage(real_data: np.ndarray, fake_data: np.ndarray, n_components: int = 5, threshold: float = 0.5) -> float:
    """
    Computes Mode Coverage by fitting a GMM on real minority data 
    and checking what fraction of modes have generated samples nearby.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(real_data)
    
    # Get cluster centers of real data
    centers = gmm.means_
    
    # Calculate minimum distance from each real mode center to the nearest fake sample
    dist_matrix = pairwise_distances(centers, fake_data)
    min_distances = dist_matrix.min(axis=1)
    
    # Count modes covered within the threshold radius
    covered_modes = np.sum(min_distances < threshold)
    coverage_score = covered_modes / n_components
    return float(coverage_score)

def evaluate_generator_performance(real_features: np.ndarray, fake_features: np.ndarray) -> dict:
    """
    Runs quantitative metrics comparing real vs synthetic feature distributions.
    """
    # 1. Mean Absolute Error between feature means (Mean Matching metric)
    mean_real = real_features.mean(axis=0)
    mean_fake = fake_features.mean(axis=0)
    mean_diff = np.mean(np.abs(mean_real - mean_fake))
    
    # 2. Mode Coverage Score
    mode_coverage = compute_mode_coverage(real_features, fake_features)
    
    metrics = {
        "Mean_Feature_Error": float(mean_diff),
        "Mode_Coverage_Score": float(mode_coverage)
    }
    return metrics

if __name__ == "__main__":
    # Dummy verification test
    print("Evaluation script loaded successfully.")
