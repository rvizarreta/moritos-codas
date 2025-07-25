import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


class JetDataset(Dataset):
    """Dataset for jet features"""
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]
        
class SimpleJetMLP(nn.Module):
    """Simple but robust MLP with strong regularization"""
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout=0.3):
        super(SimpleJetMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

def load_processed_data():
    """
    Load processed data with unique IDs and add derived features.
    Returns:
        tuple: (X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids)
    """
    # Load training data
    X_train = pd.read_csv('data/train/features/cluster_features.csv')
    y_train = np.load('data/train/labels/labels.npy')
    train_ids = np.load('data/train/ids/ids.npy')
    
    # Load validation data
    X_val = pd.read_csv('data/val/features/cluster_features.csv')
    y_val = np.load('data/val/labels/labels.npy')
    val_ids = np.load('data/val/ids/ids.npy')
    
    # Load test data
    X_test = pd.read_csv('data/test/features/cluster_features.csv')
    test_ids = np.load('data/test/ids/ids.npy')
    
    # Add derived features to all datasets
    X_train = add_derived_features(X_train)
    X_val = add_derived_features(X_val)
    X_test = add_derived_features(X_test)
    
    # Return the data
    return X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids

def add_derived_features(df):
    """
    Add all derived features to the dataframe.
    """
    df = df.copy()
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # 1. Ratio-Based Features
    # For pt_fraction_top2, we need to estimate second highest pt
    # Since we have cluster_pt_ratio = max/second_max, we can derive: second_max = max/ratio
    df['second_highest_pt'] = df['max_cluster_pt'] / (df['cluster_pt_ratio'] + eps)
    df['pt_fraction_top2'] = (df['max_cluster_pt'] + df['second_highest_pt']) / (df['total_pt'] + eps)
    
    # Assuming uniform distribution for top 3 (approximation)
    df['pt_fraction_top3'] = df['pt_fraction_top2'] + df['mean_cluster_pt'] / (df['total_pt'] + eps)
    df['pt_fraction_top3'] = df['pt_fraction_top3'].clip(upper=1.0)  # Cap at 1
    
    # Size fractions
    total_size = df['n_clusters'] * df['mean_cluster_size']
    df['size_fraction_largest'] = df['max_cluster_size'] / (total_size + eps)
    
    # Mean to max ratios
    df['mean_to_max_pt_ratio'] = df['mean_cluster_pt'] / (df['max_cluster_pt'] + eps)
    df['mean_to_max_size_ratio'] = df['mean_cluster_size'] / (df['max_cluster_size'] + eps)
    
    # 2. Concentration Metrics
    # For pt concentration, approximate sum of squares
    # Using the fact that Var = E[X²] - E[X]², we get E[X²] = Var + E[X]²
    mean_pt_squared = df['std_cluster_pt']**2 + df['mean_cluster_pt']**2
    sum_pt_squared = mean_pt_squared * df['n_clusters']
    df['pt_concentration'] = df['max_cluster_pt']**2 / (sum_pt_squared + eps)
    
    # Size concentration
    mean_size_squared = df['std_cluster_size']**2 + df['mean_cluster_size']**2
    sum_size_squared = mean_size_squared * df['n_clusters']
    df['size_concentration'] = df['max_cluster_size']**2 / (sum_size_squared + eps)
    
    # Coefficient of variation
    df['cv_cluster_pt'] = df['std_cluster_pt'] / (df['mean_cluster_pt'] + eps)
    df['cv_cluster_size'] = df['std_cluster_size'] / (df['mean_cluster_size'] + eps)
    
    # 3. Asymmetry Features
    df['pt_asymmetry'] = (df['max_cluster_pt'] - df['mean_cluster_pt']) / (df['total_pt'] + eps)
    df['size_asymmetry'] = (df['max_cluster_size'] - df['mean_cluster_size']) / (total_size + eps)
    df['spatial_asymmetry'] = np.abs(df['mean_cluster_eta'] - df['mean_cluster_phi'])
    
    # 4. Normalized Features
    df['normalized_max_pt'] = df['max_cluster_pt'] / (df['total_pt'] + eps)
    df['normalized_std_pt'] = df['std_cluster_pt'] / (df['total_pt'] + eps)
    df['clusters_per_gev'] = df['n_clusters'] / (df['total_pt'] + eps)
    
    # 5. Combined Spatial-Energy Features
    df['eta_weighted_pt'] = df['max_cluster_eta'] * df['max_cluster_pt'] / (df['total_pt'] + eps)
    df['phi_weighted_pt'] = df['max_cluster_phi'] * df['max_cluster_pt'] / (df['total_pt'] + eps)
    df['spatial_extent'] = df['max_cluster_eta'] * df['max_cluster_phi']
    
    # 6. Statistical Moments
    df['pt_skewness'] = (df['max_cluster_pt'] - df['mean_cluster_pt']) / (df['std_cluster_pt'] + eps)
    df['size_skewness'] = (df['max_cluster_size'] - df['mean_cluster_size']) / (df['std_cluster_size'] + eps)
    
    # 7. Logarithmic Features
    df['log_cluster_pt_ratio'] = np.log(df['cluster_pt_ratio'] + 1)  # log(x+1) to handle zeros
    df['log_n_clusters'] = np.log(df['n_clusters'] + 1)
    df['log_total_pt'] = np.log(df['total_pt'] + 1)
    
    # 8. Binary/Categorical Features
    df['is_single_cluster'] = (df['n_clusters'] == 1).astype(int)
    df['has_dominant_cluster'] = (df['cluster_pt_ratio'] > 3).astype(int)
    
    # PT categories (you can adjust thresholds based on your data distribution)
    df['pt_category_low'] = (df['total_pt'] < df['total_pt'].quantile(0.33)).astype(int)
    df['pt_category_high'] = (df['total_pt'] > df['total_pt'].quantile(0.67)).astype(int)
    
    # 9. Interaction Features
    df['pt_size_correlation'] = (df['normalized_max_pt'] * df['size_fraction_largest'])
    df['spatial_spread'] = np.sqrt(df['mean_cluster_eta']**2 + df['mean_cluster_phi']**2)
    df['n_clusters_squared'] = df['n_clusters']**2
    
    # 10. Inverse Features
    df['inv_n_clusters'] = 1 / (df['n_clusters'] + eps)
    df['inv_cluster_pt_ratio'] = 1 / (df['cluster_pt_ratio'] + eps)
    
    # 11. Additional useful features
    df['pt_variance_ratio'] = df['std_cluster_pt']**2 / (df['mean_cluster_pt']**2 + eps)
    df['size_variance_ratio'] = df['std_cluster_size']**2 / (df['mean_cluster_size']**2 + eps)
    
    # Energy balance indicator
    df['energy_balance'] = 1 - df['normalized_max_pt']
    
    # Relative spreads
    df['relative_eta_spread'] = df['max_cluster_eta'] / (df['mean_cluster_eta'] + eps)
    df['relative_phi_spread'] = df['max_cluster_phi'] / (df['mean_cluster_phi'] + eps)

    # Add these high-impact angular features
    df['phi_variance'] = df['max_cluster_phi']**2 - df['mean_cluster_phi']**2
    df['eta_phi_correlation'] = df['mean_cluster_eta'] * df['mean_cluster_phi']
    df['angular_asymmetry'] = (df['max_cluster_phi'] - df['mean_cluster_phi']) / (df['max_cluster_phi'] + eps)
    
    # Drop intermediate columns that were only used for calculations
    df = df.drop(columns=['second_highest_pt'], errors='ignore')
    
    # Handle any potential infinities or NaNs
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

class EnsembleJetClassifier:
    """Ensemble of simple models to reduce overfitting"""
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.scalers = []
        
    def train_single_model(self, X_train, y_train, X_val, y_val, seed=42):
        """Train a single model with different initialization"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Use RobustScaler for better generalization
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create datasets
        train_dataset = JetDataset(X_train_scaled, y_train)
        val_dataset = JetDataset(X_val_scaled, y_val)
        
        # Smaller batch size for better generalization
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Create model
        model = SimpleJetMLP(X_train.shape[1], hidden_dims=[32, 16], dropout=0.4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Training setup with L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            
            val_auc = roc_auc_score(val_labels, val_preds)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, scaler, best_val_auc
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Train ensemble of models"""
        # Use only the most important features to reduce overfitting
        important_features = ['max_cluster_phi', 'spatial_spread', 'n_clusters', 
                            'phi_weighted_pt', 'total_pt', 'mean_cluster_pt',
                            'cluster_pt_ratio', 'max_cluster_eta', 'mean_cluster_phi',
                            'spatial_extent', 'normalized_max_pt', 'pt_concentration']
        
        # Filter to available features
        available_features = [f for f in important_features if f in X_train.columns]
        X_train_filtered = X_train[available_features]
        X_val_filtered = X_val[available_features]
        self.features = available_features

        print(f"Using {len(available_features)} features")
        
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}")
            model, scaler, val_auc = self.train_single_model(
                X_train_filtered, y_train, X_val_filtered, y_val, seed=42+i*100
            )
            self.models.append(model)
            self.scalers.append(scaler)
            print(f"Model {i+1} Val AUC: {val_auc:.4f}")
        
        # Get ensemble validation score
        ensemble_preds = self.predict_proba(X_val)
        ensemble_auc = roc_auc_score(y_val, ensemble_preds)
        print(f"\nEnsemble Val AUC: {ensemble_auc:.4f}")
        
        self.features = available_features
        
    def predict_proba(self, X):
        """Get ensemble predictions"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_filtered = X[self.features]
        
        all_predictions = []
        
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X_filtered)
            dataset = JetDataset(X_scaled)
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            
            all_predictions.append(predictions)
        
        # Average predictions
        return np.mean(all_predictions, axis=0)

# Alternative: Use original BDT with conservative settings
def train_conservative_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with conservative parameters to avoid overfitting"""
    import xgboost as xgb
    
    # Use only original features
    original_features = ['n_clusters', 'total_pt', 'max_cluster_size', 'mean_cluster_size', 
                        'std_cluster_size', 'max_cluster_pt', 'mean_cluster_pt', 'std_cluster_pt',
                        'max_cluster_eta', 'max_cluster_phi', 'mean_cluster_eta', 'mean_cluster_phi',
                        'cluster_pt_ratio', 'cluster_size_ratio']
    
    X_train_orig = X_train[original_features]
    X_val_orig = X_val[original_features]
    
    # Conservative parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=1,
        reg_lambda=1,
        random_state=42
    )
    
    model.fit(X_train_orig, y_train)
    val_pred = model.predict_proba(X_val_orig)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"Conservative XGBoost Val AUC: {val_auc:.4f}")
    
    return model, original_features

if __name__ == "__main__":
    # Load data
    X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids = load_processed_data()
    
    print("Training ensemble of simple models...")
    ensemble = EnsembleJetClassifier(n_models=5)
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Generate test predictions
    test_predictions = ensemble.predict_proba(X_test)
    
    # Save predictions
    solution = pd.DataFrame({'id': test_ids, 'label': test_predictions})
    solution.to_csv('solution_ensemble_simple.csv', index=False)
    print("Predictions saved to solution_ensemble_simple.csv")
    
    # Also try conservative XGBoost
    print("\nTraining conservative XGBoost...")
    xgb_model, xgb_features = train_conservative_xgboost(X_train, y_train, X_val, y_val)
    
    test_predictions_xgb = xgb_model.predict_proba(X_test[xgb_features])[:, 1]
    solution_xgb = pd.DataFrame({'id': test_ids, 'label': test_predictions_xgb})
    solution_xgb.to_csv('solution_xgb_conservative.csv', index=False)
    print("XGBoost predictions saved to solution_xgb_conservative.csv")