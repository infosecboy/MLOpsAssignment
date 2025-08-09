#!/usr/bin/env python3
"""
Test script for the ML API
This script demonstrates how to interact with the containerized ML API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_info():
    """Test the model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction_with_features():
    """Test prediction with features object format"""
    print("Testing prediction with features object...")
    
    # Sample California housing data
    data = {
        "features": [8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction_with_array():
    """Test prediction with array format"""
    print("Testing prediction with array format...")
    
    # Sample California housing data as array
    data = [8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_multiple_predictions():
    """Test multiple predictions with different data"""
    print("Testing multiple predictions...")
    
    test_cases = [
        {
            "name": "High-value area",
            "features": [8.3252, 41.0, 6.98, 1.02, 322.0, 2.56, 37.88, -122.23]
        },
        {
            "name": "Low-value area", 
            "features": [2.3542, 30.0, 5.5, 0.8, 500.0, 3.2, 34.05, -117.15]
        },
        {
            "name": "Medium-value area",
            "features": [5.6431, 28.0, 6.2, 1.1, 280.0, 2.8, 36.77, -119.74]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test case {i}: {test_case['name']}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"features": test_case["features"]})
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Predicted value: {result['prediction'][0]:.3f}")
        else:
            print(f"  Error: {response.json()}")
        print()

if __name__ == "__main__":
    print("=== ML API Test Suite ===\n")
    
    try:
        test_health()
        test_info()
        test_prediction_with_features()
        test_prediction_with_array()
        test_multiple_predictions()
        
        print("=== All tests completed ===")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the Docker container is running on port 5000.")
    except Exception as e:
        print(f"Error: {e}")