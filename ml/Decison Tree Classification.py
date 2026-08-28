import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load Iris Dataset
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

# Display first 5 rows
print(dataset.head())

# Prepare Features and Target
X = dataset.drop('species', axis=1)
y = dataset['species']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree Model
dt = DecisionTreeClassifier(
    criterion='entropy',
    splitter='random',
    max_depth=3,
    random_state=42
)

# Train Model
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Tree Visualization
class_names_list = y.unique().astype(str).tolist()
feature_names_list = X.columns.tolist()

plt.figure(figsize=(12, 8))

plot_tree(
    dt,
    filled=True,
    feature_names=feature_names_list,
    class_names=class_names_list,
    rounded=True
)

plt.title("Decision Tree Visualization")
plt.show()
