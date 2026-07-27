from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()

# Store the features (measurements)
X = iris.data

# Store the target labels (flower species)
y = iris.target

# Display information about the dataset
print("Dataset Shape:", X.shape)
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the sizes of each dataset
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Create the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

print("\nDecision Tree model created successfully.")
# Train the Decision Tree model
model.fit(X_train, y_train)

print("Model trained successfully.")

# Train the Decision Tree model
model.fit(X_train, y_train)

print("Model trained successfully.")

# Make predictions using the test data
predictions = model.predict(X_test)

print("\nPredictions completed successfully.")

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Display a detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))