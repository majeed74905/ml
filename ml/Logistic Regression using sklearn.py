import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Prompt user to upload 'breastcancer.csv' from their local machine
print("Please upload your 'breastcancer.csv' file:")
uploaded = files.upload()

# Read the uploaded CSV
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Display data
display(df.head())

# Convert diagnosis
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features and target
# Drop non-feature columns; checking if 'Unnamed: 32' exists before dropping
cols_to_drop = [col for col in ["diagnosis", "id", "Unnamed: 32"] if col in df.columns]
X = df.drop(cols_to_drop, axis=1)
y = df["diagnosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plot
plt.imshow(cm)
plt.title("Random Forest - Breast Cancer")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()
