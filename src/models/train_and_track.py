import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data
from sklearn.metrics import r2_score


# Load and preprocess the data
df = load_dataset('california_housing')
data = preprocess_data(df)

# Unpack the preprocessed data
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Save the dataset to a file
dataset_path = "california_housing_sample.csv"
df.to_csv(dataset_path, index=False)

# Define a function to train, log, evaluate, and register models
def train_log_evaluate_and_register_model(model, model_name):
    with mlflow.start_run(run_name=model_name) as run:
        # Log the dataset as an artifact
        mlflow.log_artifact(dataset_path, artifact_path="datasets")

        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log the model with input example
        input_example = X_test.iloc[0:1]  # Use the first row of X_test as an example
        mlflow.sklearn.log_model(model, name="model", input_example=input_example)


        return rmse, model

# Train, log, and evaluate models
linear_regression_model = LinearRegression()
linear_rmse, linear_model = train_log_evaluate_and_register_model(linear_regression_model, "Linear Regression")

decision_tree_model = DecisionTreeRegressor()
decision_tree_rmse, decision_tree_model = train_log_evaluate_and_register_model(decision_tree_model, "Decision Tree")

# Select the best model based on RMSE
best_model_name = "Linear Regression" if linear_rmse < decision_tree_rmse else "Decision Tree"
best_model = linear_model if linear_rmse < decision_tree_rmse else decision_tree_model

# Save the best model as a pickle file for API use
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save model metadata
model_info = {
    'model_name': best_model_name,
    'rmse': linear_rmse if best_model_name == "Linear Regression" else decision_tree_rmse,
    'feature_names': list(X_train.columns)
}

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

# Register the best model
with mlflow.start_run(run_name=f"Best {best_model_name} Model") as run:
    input_example = X_test.iloc[0:1]  # Use the first row of X_test as an example
    mlflow.sklearn.log_model(best_model, name="model", input_example=input_example)
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=f"Best {best_model_name} Model")

print(f"Best model ({best_model_name}) registered in MLflow and saved as pickle file.")