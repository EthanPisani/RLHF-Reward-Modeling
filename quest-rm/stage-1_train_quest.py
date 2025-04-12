import os
from sklearn.linear_model import Ridge
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from safetensors.torch import load_file
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# Neural Network Architecture
# ---------------------------
class MultiObjectiveRegressor(nn.Module):
    def __init__(self, input_dim, num_attributes, hidden_dims=[512, 256], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.shared_backbone = nn.Sequential(*layers)
        self.attribute_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_attributes)
        ])
        
    def forward(self, x):
        features = self.shared_backbone(x)
        return torch.cat([head(features) for head in self.attribute_heads], dim=1)

# ---------------------------
# Training Utilities
# ---------------------------
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = nn.functional.mse_loss(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            total_loss += nn.functional.mse_loss(outputs, y_batch).item()
    return total_loss / len(dataloader)

# ---------------------------
# Argument Parsing (Maintained from original)
# ---------------------------
parser = ArgumentParser(description="Neural Probing on Precomputed Embeddings")
parser.add_argument(
    "--model_path",
    type=str,
    default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="RLHFlow/ArmoRM-Multi-Objective-Data-v0.1",
    help="Path to the dataset containing multi-objective labels (HuggingFace path or local folder)",
)
parser.add_argument(
    "--embeddings_dir",
    type=str,
    default=None,
    help="Path to the directory containing embedding files. If not provided, defaults to HOME/data/ArmoRM/embeddings/<model_name>/<dataset_name>",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Path to save the regression weights. If not provided, defaults to HOME/data/ArmoRM/regression_weights/",
)
parser.add_argument(
    "--method",
    type=str,
    default="neural",
    choices=["linear", "neural"],
    help="Method to use: 'linear' for Ridge regression or 'neural' for deep network",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of training epochs for neural method",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size for neural training",
)
args = parser.parse_args()

# Extract names from paths
args.model_name = args.model_path.split("/")[-1]
args.dataset_name = args.dataset_path.split("/")[-1]

# ---------------------------
# Configuration and Setup (Maintained from original)
# ---------------------------
attributes = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

# Set the home directory
HOME = os.path.expanduser("~")

# Define paths
if args.embeddings_dir:
    embeddings_path = args.embeddings_dir
else:
    embeddings_path = os.path.join(
        HOME, "data", "ArmoRM", "embeddings", args.model_name, args.dataset_name
    )

if args.output_dir:
    save_dir = args.output_dir
else:
    save_dir = os.path.join(HOME, "data", "ArmoRM", "regression_weights")

os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{args.model_name}_{args.dataset_name}.pt")

# ---------------------------
# Data Loading (Maintained from original)
# ---------------------------
print(f"Searching for embedding files in {embeddings_path}...")
embedding_files = sorted(glob(f"{embeddings_path}*.safetensors"))
print(f"Total embedding files found: {len(embedding_files)}")

embeddings = []
labels = []
print("Loading embeddings and labels from Safetensors files...")
for file in tqdm(embedding_files, desc="Loading embeddings"):
    data = load_file(file)
    embeddings.append(data["embeddings"])
    labels.append(data["labels"])

embeddings = torch.cat(embeddings, dim=0).float().numpy()
labels = torch.cat(labels, dim=0).float().numpy()

print(f"Total embeddings loaded: {embeddings.shape[0]}")
print(f"Total labels loaded: {labels.shape[0]}")

# ---------------------------
# Data Splitting (Maintained from original)
# ---------------------------
print("Splitting data into training and validation sets...")
X_train, X_val, Y_train, Y_val = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# ---------------------------
# Training Selection
# ---------------------------
if args.method == "linear":
    # Original Ridge regression implementation
    print("Using linear Ridge regression method...")
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    df = pd.DataFrame(columns=["attribute", "alpha", "loss"])
    for attr_idx in tqdm(range(Y_train.shape[1]), desc="Attributes"):
        y_train = Y_train[:, attr_idx]
        valid_mask_train = ~np.isnan(y_train)
        y_train_filtered = y_train[valid_mask_train]
        X_train_filtered = X_train[valid_mask_train]

        y_val = Y_val[:, attr_idx]
        valid_mask_val = ~np.isnan(y_val)
        y_val_filtered = y_val[valid_mask_val]
        X_val_filtered = X_val[valid_mask_val]

        for alpha in tqdm(alphas, desc=f"Alpha for attribute {attr_idx}", leave=False):
            clf = Ridge(alpha=alpha, fit_intercept=False)
            clf.fit(X_train_filtered, y_train_filtered)
            pred = clf.predict(X_val_filtered)
            loss = mean_squared_error(y_val_filtered, pred)
            df = df._append(
                {"attribute": attr_idx, "alpha": alpha, "loss": loss}, ignore_index=True
            )

    best_alphas = df.loc[df.groupby("attribute")["loss"].idxmin()]
    print("Best alphas selected for each attribute:")
    print(best_alphas)

    weights = []
    for index, row in best_alphas.iterrows():
        attr_idx = int(row["attribute"])
        best_alpha = row["alpha"]
        clf = Ridge(alpha=best_alpha, fit_intercept=False)
        
        y_train = Y_train[:, attr_idx]
        valid_mask_train = ~np.isnan(y_train)
        clf.fit(X_train[valid_mask_train], y_train[valid_mask_train])
        weights.append(clf.coef_)

    weights = np.stack(weights)

else:
    # Neural network implementation
    print("Using neural network method...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Filter out samples with any NaN labels
    print("Filtering out samples with NaN labels...")
    valid_samples = ~np.isnan(Y_train).any(axis=1)
    X_train_filtered = X_train[valid_samples]
    Y_train_filtered = Y_train[valid_samples]

    valid_samples_val = ~np.isnan(Y_val).any(axis=1)
    X_val_filtered = X_val[valid_samples_val]
    Y_val_filtered = Y_val[valid_samples_val]

    # Create datasets and loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_filtered), 
        torch.FloatTensor(Y_train_filtered)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_filtered), 
        torch.FloatTensor(Y_val_filtered)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = MultiObjectiveRegressor(
        input_dim=X_train.shape[1],
        num_attributes=len(attributes),
        hidden_dims=[1024, 512, 256],
        dropout=0.3
    ).to(device)

    # Training configuration
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    early_stopper = EarlyStopper(patience=5)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        
        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))

    # Extract weights
    weights = []
    for head in model.attribute_heads:
        head_weights = []
        for layer in head:
            if isinstance(layer, nn.Linear):
                head_weights.append(layer.weight.detach().cpu().numpy())
        combined = head_weights[1] @ head_weights[0]  # W2 * W1
        weights.append(combined)

    shared_weights = model.shared_backbone[-4].weight.detach().cpu().numpy()
    weights = np.stack([w @ shared_weights for w in weights])

# ---------------------------
# Save Results (Maintained from original)
# ---------------------------
torch.save({"weight": torch.from_numpy(weights)}, save_path)
print(f"Saved regression weights to {save_path}")

# Print final weight statistics
print("\nFinal weight statistics:")
for i, attr in enumerate(attributes):
    print(f"{attr}: mean={weights[i].mean():.4f}, std={weights[i].std():.4f}")