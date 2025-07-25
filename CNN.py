# -----------------------------
# Step 1: Imports
# -----------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import random

# -----------------------------
# Step 2: Set Random Seeds
# -----------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# -----------------------------
# Step 3: Simulate Dummy Data
# -----------------------------
n_samples = 5000
n_features = 30

X = np.random.normal(0, 1, (n_samples, n_features)).astype(np.float32)
y = np.random.binomial(1, p=0.5, size=n_samples).astype(np.float32)

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)

# -----------------------------
# Step 4: Dataset Class
# -----------------------------
class JetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Step 5: CNN Model
# -----------------------------
class JetCNN1D(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super(JetCNN1D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, F)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.squeeze(-1)  # Logits

# -----------------------------
# Step 6: Ensemble Classifier
# -----------------------------
class EnsembleJetClassifier:
    def __init__(self, n_models=5, batch_size=128, learning_rate=1e-3, device='cpu'):
        self.n_models = n_models
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.models = []
        self.scalers = []

    def train_single_model(self, X_train, y_train, X_val, y_val, seed):
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Datasets and loaders
        train_dataset = JetDataset(X_train, y_train)
        val_dataset = JetDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Model and optimizer
        model = JetCNN1D(input_dim=X_train.shape[1], dropout=0.4).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_auc = 0.0
        best_model_state = None

        for epoch in range(15):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    logits = model(xb)
                    all_logits.extend(torch.sigmoid(logits).cpu().numpy())
                    all_labels.extend(yb.numpy())

            auc = roc_auc_score(all_labels, all_logits)
            if auc > best_auc:
                best_auc = auc
                best_model_state = model.state_dict()

        # Load best weights
        model.load_state_dict(best_model_state)
        return model, scaler, best_auc

    def fit(self, X_train, y_train, X_val, y_val):
        self.models = []
        self.scalers = []

        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}")
            model, scaler, auc = self.train_single_model(
                X_train, y_train, X_val, y_val, seed=42 + i * 100
            )
            print(f"Model {i+1} AUC: {auc:.4f}")
            self.models.append(model)
            self.scalers.append(scaler)

    def predict_proba(self, X):
        preds = []
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            dataset = JetDataset(X_scaled, np.zeros(len(X)))
            loader = DataLoader(dataset, batch_size=self.batch_size)
            model.eval()
            probs = []
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    logits = model(xb)
                    probs.extend(torch.sigmoid(logits).cpu().numpy())
            preds.append(np.array(probs))
        return np.mean(preds, axis=0)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

# -----------------------------
# Step 7: Run It All
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ensemble = EnsembleJetClassifier(n_models=3, device=device)
ensemble.fit(X_train, y_train, X_val, y_val)

# Test predictions
probs = ensemble.predict_proba(X_test)
preds = ensemble.predict(X_test)

# Final test AUC
test_auc = roc_auc_score(y_test, probs)
print(f"Test AUC: {test_auc:.4f}")