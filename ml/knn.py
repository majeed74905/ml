import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
 
# Load 
dataset = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv') 
x = dataset.drop('species', axis=1).values 
 
# Corrected typo: use the same variable name 
kmeans = KMeans(n_clusters=3, random_state=42) # Changed from 30 to 3 for iris
kmeans.fit(x) 
 
cl = kmeans.labels_ 
ce = kmeans.cluster_centers_ 
 
print("type of x:", type(x), "shape of x:", x.shape) 
print("type of cl:", type(cl), "shape of cl:", cl.shape) 
print("type of ce:", type(ce), "shape of ce:", ce.shape) 
 
# Plotting 
plt.figure(figsize=(10, 6)) 
plt.scatter(x[:, 0], x[:, 1], c=cl, cmap='viridis', label='Data Points') 
plt.scatter(ce[:, 0], ce[:, 1], c='red', marker='x', s=200, label='Centroids') 
plt.xlabel("Sepal Length (cm)") 
plt.ylabel("Sepal Width (cm)") 
plt.title("K-Means Clustering") 
plt.legend() 
plt.show()
