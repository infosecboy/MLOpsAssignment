from data.load_data import load_dataset
from data.preprocess import preprocess_data

def main():
    # Load the California Housing dataset
    print("Loading California Housing dataset...")
    df = load_dataset('california_housing')
    
    # Preprocess the data
    print("\nPreprocessing data...")
    processed_data = preprocess_data(df)
    
    print("\nData processing complete!")
    print(f"Training set shape: {processed_data['X_train'].shape}")
    print(f"Test set shape: {processed_data['X_test'].shape}")
    print("\nFeatures:")
    print(processed_data['X_train'].columns.tolist())

if __name__ == "__main__":
    main()