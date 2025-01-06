import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def generate_synthetic_data(n_samples=1000, n_features=7, n_classes=5, test_size=0.2, random_state=42):
    """Generate synthetic data with well-separated classes."""
    np.random.seed(random_state)
    
    # Generate more samples initially to account for balancing
    initial_samples = n_samples * 2
    
    # Generate base features with larger scale for better separation
    X = np.random.randn(initial_samples, n_features) * 2
    
    # Create more distinct non-linear relationships
    X[:, 0] = 3 * np.sin(X[:, 0]) + np.random.randn(initial_samples) * 0.1
    X[:, 1] = 2 * np.exp(X[:, 1] / 3) + np.random.randn(initial_samples) * 0.1
    X[:, 2] = X[:, 0] * X[:, 1] + np.random.randn(initial_samples) * 0.1
    X[:, 3] = 2 * np.square(X[:, 3]) + np.random.randn(initial_samples) * 0.1
    X[:, 4] = np.tanh(X[:, 4]) * 3 + np.random.randn(initial_samples) * 0.1
    X[:, 5] = np.cos(X[:, 5]) * 2 + np.random.randn(initial_samples) * 0.1
    X[:, 6] = np.sign(X[:, 6]) * np.log(np.abs(X[:, 6]) + 1) + np.random.randn(initial_samples) * 0.1
    
    # Generate target classes with more distinct decision boundaries
    logits = np.zeros((initial_samples, n_classes))
    logits[:, 0] = 2 * np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 4]
    logits[:, 1] = X[:, 2] * X[:, 3] - np.square(X[:, 1]) + X[:, 5]
    logits[:, 2] = np.exp(X[:, 0]/3) - np.sin(X[:, 2] * X[:, 3]) + X[:, 6]
    logits[:, 3] = np.tanh(X[:, 4] + X[:, 5]) + np.cos(X[:, 0] * X[:, 1])
    logits[:, 4] = np.sin(X[:, 6]) + np.exp(X[:, 2]/3) - np.cos(X[:, 3])
    
    # Add class-specific bias to balance classes
    class_biases = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    logits += class_biases[np.newaxis, :]
    
    # Convert to probabilities and select class
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    y = np.argmax(probs, axis=1)
    
    # Ensure we have enough samples per class after balancing
    min_samples = n_samples // n_classes
    balanced_indices = []
    for class_idx in range(n_classes):
        class_indices = np.where(y == class_idx)[0]
        if len(class_indices) < min_samples:
            # If we don't have enough samples, sample with replacement
            class_indices = np.random.choice(class_indices, min_samples, replace=True)
        else:
            # If we have enough samples, sample without replacement
            class_indices = np.random.choice(class_indices, min_samples, replace=False)
        balanced_indices.extend(class_indices)
    
    # Ensure exact number of samples
    if len(balanced_indices) > n_samples:
        balanced_indices = np.random.choice(balanced_indices, n_samples, replace=False)
    
    X = X[balanced_indices]
    y = y[balanced_indices]
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create DataFrames
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Split with stratification
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['target']
    )
    
    return train_df, val_df

def main():
    # Create input_data directory if it doesn't exist
    os.makedirs('input_data', exist_ok=True)
    
    # Generate data with explicit sizes
    train_df, val_df = generate_synthetic_data(
        n_samples=1000,
        n_features=7,  # Updated number of features
        n_classes=5,   # Updated number of classes
        test_size=0.2,
        random_state=42
    )
    
    # Print detailed information
    print(f"\nData shapes:")
    print(f"Training data: {train_df.shape}")
    print(f"Validation data: {val_df.shape}")
    print("\nFeature names:")
    print(train_df.columns.tolist())
    print("\nClass distribution in training set:")
    print(train_df['target'].value_counts(normalize=True).sort_index())
    print("\nClass distribution in validation set:")
    print(val_df['target'].value_counts(normalize=True).sort_index())
    
    # Save to CSV files
    train_df.to_csv('input_data/train.csv', index=False)
    val_df.to_csv('input_data/val.csv', index=False)
