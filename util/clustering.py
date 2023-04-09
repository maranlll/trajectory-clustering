from sklearn.cluster import DBSCAN
import numpy as np
import hdbscan
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import math
import kmedoids

def DBSCAN_cluster(trajectory_list, distance_matrix, eps, min_samples, metric):
    print("start dbscan clustering")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(distance_matrix)

    num_cluster = np.max(dbscan.labels_) + 1
    result = [[] for i in range(num_cluster)]
    noise = []

    # post
    size = len(trajectory_list)
    for i in range(size):
        label = dbscan.labels_[i]
        if (label == -1):
            noise.append(trajectory_list[i])
        else:
            result[label].append(trajectory_list[i])

    print("label size: %s" % num_cluster)
    print("noise size: %s" % len(noise))
    return num_cluster, result, noise

def HDBSCAN_cluster(trajectory_list, distance_matrix, min_cluster_size, min_samples, metric):
    print("start hdbscan clustering")
    dbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, memory="hdbscan_cache")
    dbscan.fit(distance_matrix)

    num_cluster = np.max(dbscan.labels_) + 1
    result = [[] for i in range(num_cluster)]
    noise = []
    
    silhoutte_score = metrics.silhouette_score(distance_matrix, dbscan.labels_)
    print("Silhoutte Coefficient : %.3f" % silhoutte_score)

    # post
    size = len(trajectory_list)
    for i in range(size):
        label = dbscan.labels_[i]
        if (label == -1):
            noise.append(trajectory_list[i])
        else:
            result[label].append(trajectory_list[i])

    print("label size: %s" % num_cluster)
    print("noise size: %s" % len(noise))
    return num_cluster, result, noise

# GMM equal length
def GMM_cluster(trajectory_list, distance_matrix, num_cluster, iter):
    gmm = GaussianMixture(n_components=num_cluster, max_iter=iter)
    # print(trajectory_list)
    labels = gmm.fit_predict(trajectory_list)
    
    result = [[] for i in range(num_cluster)]
    size = len(trajectory_list)
    for i in range(size):
        label = labels[i]
        result[label].append(trajectory_list[i])
        
    silhoutte_score = metrics.silhouette_score(distance_matrix, labels, sample_size=1000)
    print("Silhoutte Coefficient : %.3f" % silhoutte_score)
    return result
    
def test_kmedoids_cluster(trajectory_list, distance_matrix, k, iter_max):
    print("start kmedoids clustering")
    m, n = np.shape(distance_matrix)
    np.fill_diagonal(distance_matrix, math.inf)
    if k > n:
        raise Exception("too many clusters")
    
    core = np.arange(n)
    np.random.shuffle(core)
    core = np.sort(core[: k])
    
    core_new = np.copy(core)
    
    result = [[] for i in range(k)]
    
    for iter in range(iter_max):
        closer = np.argmin(distance_matrix[:, core], axis=1)
        
        for i in range(k):
            result[i] = np.where(closer==i)[0]
            
        for i in range(k):
            size = len(result[i])
            dis = np.delete(distance_matrix[np.ix_(result[i], result[i])], [x * size + x for x in range(size)]).reshape(size, size - 1)
            closer = np.mean(dis, axis=1)
            center = np.argmin(closer)
            core_new[i] = result[i][center]
            
        core_new = np.sort(core_new)
        
        if np.array_equal(core, core_new):
            print("iter times: %s" % iter)
            break
        core = np.copy(core_new)
    else:
        closer = np.argmin(distance_matrix[:, core], axis=1)
        for i in range(k):
            result[i] = np.where(closer==i)[0]
        print("iter times: %s" % iter)
            
    np.fill_diagonal(distance_matrix, 0)
    
    for i in range(k):
        result[i] = [trajectory_list[t] for t in result[i]]
        
    core = [trajectory_list[t] for t in core]
        
    return result, core

def kmedoids_cluster(trajectory_list, distance_matrix, k, method='fasterpam'):
    print("start kemdoids clustering")
    pam = kmedoids.KMedoids(k, method=method)
    ret = pam.fit(distance_matrix)
    result = [[] for i in range(k)]
    size = len(trajectory_list)
    for i in range(size):
        label = ret.labels_[i]
        result[label].append(trajectory_list[i])
    return result
            
    