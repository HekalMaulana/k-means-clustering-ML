import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans

wcss = []
# Pengujian dengan menggunakan elbow_method
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state= 42)
#     kmeans.fit(X=X)
#     WCSS.append(kmeans.inertia_)
# plt.plot(range(1, 11), WCSS )
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
    