from sklearn.datasets import load_iris, fetch_california_housing
import pandas as pd
import os

def load_dataset(dataset_name='iris'):
    """
    Load either the Iris or California Housing dataset
    Args:
        dataset_name (str): Either 'iris' or 'california_housing'
    Returns:
        pandas DataFrame: The loaded dataset
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name.lower() == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name.lower() == 'california_housing':
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        raise ValueError("Dataset must be either 'iris' or 'california_housing'")
    
    # Save raw data
    df.to_csv(os.path.join(data_dir, f'{dataset_name}_raw.csv'), index=False)
    return df

if __name__ == "__main__":
    # You can change this to 'california_housing' if you prefer
    df = load_dataset('iris')
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())