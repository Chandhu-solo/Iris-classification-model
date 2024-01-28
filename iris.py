import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from tensorflow.keras import layers

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Using scikit-learn RandomForestClassifier
# -------------------------------------------

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 2. Using TensorFlow for Neural Network Classification
# -----------------------------------------------------

# One-hot encode the target variable
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=3)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=3)

# Initialize the model
model_nn = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_nn.fit(X_train_scaled, y_train_one_hot, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions
y_pred_nn_one_hot = model_nn.predict(X_test_scaled)
y_pred_nn = np.argmax(y_pred_nn_one_hot, axis=1)

# Evaluate the model
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print("\nNeural Network Classifier:")
print(f"Accuracy: {accuracy_nn}")
print("Classification Report:")
print(classification_report(y_test, y_pred_nn))
