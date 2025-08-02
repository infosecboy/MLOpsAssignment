from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the dataset: scale features and split into train/test sets
    Args:
        df (pandas DataFrame): The input dataset
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
    Returns:
        dict: Dictionary containing X_train, X_test, y_train, y_test, and scaler
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame to keep feature names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Save processed data
    processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join(processed_dir, 'train.csv'), index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join(processed_dir, 'test.csv'), index=False
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }