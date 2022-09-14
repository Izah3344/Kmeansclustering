import pandas as pd 
import numpy as np 
#import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.cluster import KMeans

file = "mall_customer.csv"
df = pd.read_csv(file)

features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'],  );

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=25, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5);
