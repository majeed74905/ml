# ==========================
# K-Nearest Neighbor (KNN)
# ==========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load Iris Dataset
iris = datasets.load_iris()

# Create DataFrame
df = pd.DataFrame(
    data=np.c_[iris.data, iris.target],
    columns=iris.feature_names + ['target']
)

print(df.head())

# Visualization
sns.FacetGrid(df, hue="target", height=6).map(
    plt.scatter,
    "petal length (cm)",
    "petal width (cm)"
).add_legend()

plt.title("Iris Dataset Visualization")
plt.show()

# ===================================
# K-Nearest Neighbor (KNN)
# ===================================

print("\n========== KNN ==========\n")

model = KNeighborsClassifier(n_neighbors=6)

model.fit(iris.data, iris.target)

predicted = model.predict(iris.data)
expected = iris.target

print("Model Accuracy:", model.score(iris.data, iris.target))

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(expected, predicted))

# ===================================
# Naive Bayes
# ===================================

print("\n========== Naive Bayes ==========\n")

model = GaussianNB()

model.fit(iris.data, iris.target)

predicted = model.predict(iris.data)
expected = iris.target

print("Model Accuracy:", model.score(iris.data, iris.target))

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(expected, predicted))
