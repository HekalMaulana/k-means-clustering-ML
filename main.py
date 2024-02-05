import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans

WCSS = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state= 42)
    kmeans.fit(X=X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1, 11), WCSS )
plt.title("ELBOW METHOD")
plt.xlabel("Numbers of cluster")
plt.ylabel("WCSS")
plt.show()