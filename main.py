import math
import random

# --- 1. Data Generation (Simulating "Big Data" concept in a small scale) ---
# A simple 2D dataset for binary classification
# Each data point is [feature1, feature2, class_label]
# Class 0: points around (1,1), Class 1: points around (5,5)
def generate_synthetic_data(num_samples=50):
    data = []
    for _ in range(num_samples // 2):
        # Class 0 data points
        x = random.uniform(0, 2)
        y = random.uniform(0, 2)
        data.append([x, y, 0])
        # Class 1 data points
        x = random.uniform(4, 6)
        y = random.uniform(4, 6)
        data.append([x, y, 1])
    random.shuffle(data)
    return data

# --- 2. Core ML Algorithm: K-Nearest Neighbors (KNN) from scratch ---
# This demonstrates a simple "learning" process based on data proximity,
# a core idea in Machine Learning.

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points (features only)."""
    distance = 0
    # Assuming features are the first elements, label is the last
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)

def get_neighbors(training_data, test_point_features, k):
    """Finds the k nearest neighbors to a test point in the training data."""
    distances = []
    for train_point in training_data:
        train_point_features = train_point[:-1] # Exclude the label
        dist = euclidean_distance(test_point_features, train_point_features)
        distances.append((train_point, dist))
    distances.sort(key=lambda x: x[1]) # Sort by distance
    neighbors = [item[0] for item in distances[:k]]
    return neighbors

def predict_classification(training_data, test_point_features, k):
    """Predicts the class label for a test point using KNN."""
    neighbors = get_neighbors(training_data, test_point_features, k)
    output_labels = [neighbor[-1] for neighbor in neighbors] # Get labels of neighbors
    
    # Count votes for each class among neighbors
    class_votes = {}
    for label in output_labels:
        class_votes[label] = class_votes.get(label, 0) + 1
    
    # Find the class with the most votes (majority class)
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0] # Return the label of the majority class

# --- 3. Model Training and Evaluation (Illustrating ML workflow) ---

if __name__ == "__main__":
    print("Yapay Zeka ve Makine Öğrenimi Araştırma Özeti: Basit Bir KNN Örneği")
    print("AI and Machine Learning Research Summary: A Simple KNN Example\n")

    # Generate our synthetic dataset, simulating data collection
    dataset = generate_synthetic_data(num_samples=100)
    print(f"Generated {len(dataset)} data points.")

    # Split data into training and testing sets (80% train, 20% test)
    # This simulates how models are evaluated in real ML scenarios.
    train_size = int(len(dataset) * 0.8)
    training_set = dataset[:train_size]
    test_set = dataset[train_size:]

    print(f"Training set size: {len(training_set)}")
    print(f"Test set size: {len(test_set)}\n")

    k_value = 5 # Number of neighbors to consider for KNN

    correct_predictions = 0
    total_predictions = len(test_set)

    print(f"Making predictions on the test set with k={k_value}...")
    for i, test_point in enumerate(test_set):
        # The features for prediction are all but the last element (the label)
        features_to_predict = test_point[:-1]
        actual_label = test_point[-1]

        # --- This is where the "AI/ML" prediction happens ---
        # The KNN algorithm uses the 'learned' patterns (the training data) to classify new data.
        predicted_label = predict_classification(training_set, features_to_predict, k_value)
        # ---------------------------------------------------

        print(f"Test point {i+1}: Features={features_to_predict}, Actual={actual_label}, Predicted={predicted_label}")
        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\n--- Prediction Results ---")
    print(f"Total test points: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nThis example demonstrates a basic Machine Learning workflow:")
    print("1. Data generation/collection (simulated).")
    print("2. A simple learning algorithm (K-Nearest Neighbors).")
    print("3. Training (data is 'learned' by the algorithm).")
    print("4. Prediction on new data.")
    print("5. Evaluation of the model's performance.")
    print("This reflects the core ideas of AI/ML research discussed in the article.")
