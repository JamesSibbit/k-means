from sklearn.cluster import KMeans

def k_means_clustering(data):
    #Now carry out k-means clustering
    k_mean = KMeans(n_clusters = 2)
    return k_mean.fit(data)
