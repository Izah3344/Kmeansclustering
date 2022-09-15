#Import Packages

import pandas as pd # working with data
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from sklearn.cluster import KMeans # K-means algorithm


df = pd.read_csv('/content/mall_customer.csv')
df.head()
features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]
X
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'],  );

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=25, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5);
