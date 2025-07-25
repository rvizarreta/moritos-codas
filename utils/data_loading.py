import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data_processing import load_images, anti_kt_clustering

def process_jet_to_clusters(image, R=0.4, pt_min=0.1, max_clusters=2):
    """Process a single jet image into anti-kt clusters.
    
    Args:
        image (np.ndarray): Jet image array
        R (float): Jet radius parameter
        pt_min (float): Minimum pT threshold
        max_clusters (int): Maximum number of clusters to keep
        
    Returns:
        np.ndarray: Array of cluster features [pt, eta, phi] for top N clusters by pT
    """
    # Get clusters from anti-kt algorithm
    clusters = anti_kt_clustering(image, R=R, pt_min=pt_min)
    
    # Convert clusters to feature array
    cluster_features = []
    for cluster in clusters:
        cluster = np.array(cluster)
        pt = np.sum(cluster[:, 2])
        eta = np.mean(cluster[:, 0])  # eta is first column
        phi = np.mean(cluster[:, 1])  # phi is second column
        features = np.array([pt, eta, phi])
        cluster_features.append(features)
    
    # Convert to numpy array
    cluster_features = np.array(cluster_features)
    
    if len(cluster_features) == 0:
        # If no clusters found, return array of zeros
        return np.zeros((max_clusters, 3))
    
    # Sort clusters by pT (first column) in descending order
    pt_order = np.argsort(-cluster_features[:, 0])
    cluster_features = cluster_features[pt_order]
    
    # Take top N clusters
    if len(cluster_features) > max_clusters:
        cluster_features = cluster_features[:max_clusters]
    elif len(cluster_features) < max_clusters:
        # Pad with zeros if we have fewer than max_clusters
        padding = np.zeros((max_clusters - len(cluster_features), 3))
        cluster_features = np.vstack([cluster_features, padding])
    
    return cluster_features

class JetClusterDataset(Dataset):
    def __init__(self, images, labels=None, R=0.4, pt_min=0.1, max_clusters=10):
        """Initialize the dataset.
        
        Args:
            images (np.ndarray): Array of jet images
            labels (np.ndarray): Array of labels
            R (float): Jet radius parameter
            pt_min (float): Minimum pT threshold
            max_clusters (int): Maximum number of clusters to keep per jet
        """
        self.images = images
        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = torch.zeros(len(images))
        self.R = R
        self.pt_min = pt_min
        self.max_clusters = max_clusters
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Process image to clusters
        clusters = process_jet_to_clusters(
            self.images[idx], 
            self.R, 
            self.pt_min,
            self.max_clusters
        )
        
        # Convert to tensor
        clusters = torch.FloatTensor(clusters)  # Shape: [max_clusters, 3]
        
        return clusters, self.labels[idx]

def get_dataloaders(batch_size=32, R=0.4, pt_min=0.1, max_clusters=10, num_workers=4):
    """Create dataloaders for train, validation, and test sets.
    
    Args:
        batch_size (int): Batch size for the dataloaders
        R (float): Jet radius parameter
        pt_min (float): Minimum pT threshold
        max_clusters (int): Maximum number of clusters to keep per jet
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load all data using the existing function
    X_train_images, y_train, train_ids, X_val_images, y_val, val_ids, X_test_images, test_ids = load_images()
    
    # Create datasets
    train_dataset = JetClusterDataset(X_train_images, y_train, R=R, pt_min=pt_min, max_clusters=max_clusters)
    val_dataset = JetClusterDataset(X_val_images, y_val, R=R, pt_min=pt_min, max_clusters=max_clusters)
    test_dataset = JetClusterDataset(X_test_images, R=R, pt_min=pt_min, max_clusters=max_clusters)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 