import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans

wcss = []
# Pengujian dengan menggunakan elbow_method
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state= 42)
    kmeans.fit(X=X)
    wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss )
# plt.title("ELBOW METHOD")
# plt.xlabel("Numbers of cluster")
# plt.ylabel("WCSS")
# plt.show()

# Pengujian mencari jumlah cluster yang tepat dengan menggunakan silhouette_score
# from sklearn.metrics import silhouette_score
# for i in range(2, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     silhouette_avg = silhouette_score(X, kmeans.fit_predict(X))
#     wcss.append(silhouette_avg)
#     print(f"For n_clusters = {i} The average silhouette_score is : {silhouette_avg}")

# ideal_silhouette_score_max = max(wcss)
# ideal_silhouette_score = wcss.index(ideal_silhouette_score_max) + 2
# print(f"Ideal Number Cluster : {ideal_silhouette_score}")

# Melatih k-means dengan dataset
ideal_num_cluster = 5
kmeans = KMeans(n_clusters=ideal_num_cluster, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

# random color
colors = np.random.rand(ideal_num_cluster, 3)

# Visualisasi k-means
for cluster_num in range(ideal_num_cluster):
    plt.scatter(X[y_kmeans == cluster_num, 0], X[y_kmeans == cluster_num, 1], s=100, c=[colors[cluster_num]], label=f"Cluster {cluster_num + 1}")
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
    