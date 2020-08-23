from k_means import k_means_clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import norm

#Create some random data using normal dist and plot to see seperation

mu_one = float(input("Enter mean for first normal cluster: "))
mu_two = float(input("Enter mean for second normal cluster: "))
var_one = float(input("Enter variance for first normal cluster: "))
var_two = float(input("Enter variance for second normal cluster: "))

X1 = np.array(norm.rvs(loc=mu_one, scale =var_one, size=100))
Y1 = np.array(norm.rvs(loc=mu_one, scale =var_one, size=100))
X2 = np.array(norm.rvs(loc=mu_two, scale =var_two, size=100))
Y2 = np.array(norm.rvs(loc=mu_two, scale =var_two, size=100))

data_zero = np.vstack((X1,Y1)).T
data_one = np.vstack((X2,Y2)).T
data = np.vstack((data_zero,data_one))

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(data_zero[:,0], data_zero[:, 1], label="Class 1")
ax1.scatter(data_one[:,0], data_one[:, 1], label="Class 2")
plt.legend(loc='upper left')
plt.show()

k_mean = k_means_clustering(data)
centroids = k_mean.cluster_centers_

print("K-means centroid output is as follows.")
print("Centroid of class one is x="+str(centroids[0][0])+", y="+str(centroids[0][1]))
print("Centroid of class two is x="+str(centroids[1][0])+", y="+str(centroids[1][1]))
print("Now enter a test sample.")

value_x = float(input("Enter x value of data point: "))
value_y = float(input("Enter y value of data point: "))

test_sample = np.array([value_x, value_y]).reshape(1,-1)
result = k_mean.predict(test_sample)

print("Value belongs to cluster "+str(result[0]+1))
