import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def generate_synthetic_data(n_samples=1000, n_features=4, n_classes=3, test_size=0.2, random_state=42):
    """Generate synthetic data for classification with some non-linear relationships."""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear relationships
    X[:, 0] = np.sin(X[:, 0]) + np.random.randn(n_samples) * 0.1
    X[:, 1] = np.exp(X[:, 1] / 2) + np.random.randn(n_samples) * 0.1
    X[:, 2] = X[:, 0] * X[:, 1] + np.random.randn(n_samples) * 0.1
    X[:, 3] = np.square(X[:, 3]) + np.random.randn(n_samples) * 0.1
    
    # Generate target classes based on non-linear combinations
    logits = np.zeros((n_samples, n_classes))
    logits[:, 0] = np.sin(X[:, 0]) + np.cos(X[:, 1])
    logits[:, 1] = X[:, 2] * X[:, 3] - np.square(X[:, 1])
    logits[:, 2] = np.exp(X[:, 0]/2) - np.sin(X[:, 2] * X[:, 3])
    
    # Convert to probabilities and select class
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    y = np.argmax(probs, axis=1)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create DataFrames
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Split into train and validation sets
    mask = np.random.rand(len(df)) < (1 - test_size)
    train_df = df[mask]
    val_df = df[~mask]
    
    return train_df, val_df

def main():
    # Create input_data directory if it doesn't exist
    os.makedirs('input_data', exist_ok=True)
    
    # Generate data
    train_df, val_df = generate_synthetic_data(
        n_samples=1000,
        n_features=4,
        n_classes=3,
        test_size=0.2,
        random_state=42
    )
    
    # Save to CSV files
    train_df.to_csv('input_data/train.csv', index=False)
    val_df.to_csv('input_data/val.csv', index=False)
    
    print(f"Generated {len(train_df)} training samples and {len(val_df)} validation samples")
    print("\nFeature statistics:")
    print(train_df.describe())
    print("\nClass distribution in training set:")
    print(train_df['target'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
