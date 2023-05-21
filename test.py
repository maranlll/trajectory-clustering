# %%
import util.plot as plt
import util.io as io
import util.similarlity as dis
import util.clustering as cluster
from util.timer import Timer
import numpy as np
import util.dataPreparation as data
from tqdm import tqdm
from scipy.sparse import lil_matrix

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

eps = 20
min_cluster_size = 10
min_samples = 10
threshold = 30
iter_max = 2000

pkl_file = [
    "data/boat/pkl_clean/AIS_TYP_ID_30_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_31_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_32_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_33_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_35_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_36_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_37_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_50_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_51_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_53_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_54_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_55_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_59_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_69_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_89_points.pkl",
    "data/boat/pkl_clean/all_points.pkl",
]

t = Timer()

# filename = 'data/1/cross.mat'
# trajectory_list = io.read_from_mat(filename)
filename = pkl_file[17]
trajectory_list = io.load(filename)

# t.start("start data preparation")
# # for i in tqdm(range(len(trajectory_list))):
# #     trajectory_list[i] = data.shorten_by_mdl(trajectory_list[i])
# trajectory_list = data.shorten_by_mdl_all_multiprocecss(trajectory_list)
# t.end("end data preparation")

plt.figure('raw')
plt.xlim(100, 1000)
plt.ylim(100, 700)
for trajectory in trajectory_list:
    plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.show()

# %%
# t.start("start generate distance_matrix, trajectory size: %s" % len(trajectory_list))
# distance_matrix = dis.generate_distance_matrix(trajectory_list, threshold)
# distance_matrix = dis.generate_dense_distance_matrix(trajectory_list)
# t.end("end generate distance_matrix")

# io.save(distance_matrix, "data/boat/pkl_clean/distance_matrix.pkl")

distance_matrix = io.load("data/boat/pkl_clean/distance_matrix.pkl")

t.start()
print("start mds")
tmp = cluster.calculate_mds(distance_matrix, 2, "precomputed")
t.end("end mds")

plt.figure('mds')
plt.scatter(tmp[:, 0], tmp[:, 1])
plt.show()

silhoutte_score_list1 = []
silhoutte_score_list2 = []
x_label = []

for i in range(5, 40, 5):
    t.start("DBSCAN start clustering, eps: %s" % i)
    num_cluster1, result1, noise1, labels1, silhoutte_score1 = cluster.DBSCAN_cluster(trajectory_list, tmp, i, min_samples, metric="euclidean")
    num_cluster2, result2, noise2, labels2, silhoutte_score2 = cluster.DBSCAN_cluster(trajectory_list, distance_matrix, i, min_samples, metric="precomputed")
    t.end("end clustering")
    x_label.append(i)
    silhoutte_score_list1.append(silhoutte_score1)
    silhoutte_score_list2.append(silhoutte_score2)
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels1, cmap='coolwarm')
    plt.show()
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels2, cmap='coolwarm')
    plt.show()

plt.figure('silhoutte_score')
plt.plot(x_label, silhoutte_score_list1, label="mds1")
plt.plot(x_label, silhoutte_score_list2, label="mds2")
plt.show()

silhoutte_score_list1 = []
silhoutte_score_list2 = []
x_label = []
for i in range(5, 40, 5):
    t.start("HDBSCAN start clustering, min_cluster_size: %s" % i)
    num_cluster1, result1, noise1, labels1, silhoutte_score1 = cluster.HDBSCAN_cluster(trajectory_list, tmp, i, min_samples, metric="euclidean")
    num_cluster2, result2, noise2, labels2, silhoutte_score2 = cluster.HDBSCAN_cluster(trajectory_list, distance_matrix, i, min_samples, metric="precomputed")
    t.end("end clustering")
    x_label.append(i)
    silhoutte_score_list1.append(silhoutte_score1)
    silhoutte_score_list2.append(silhoutte_score2)
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels1, cmap='coolwarm')
    plt.show()
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels2, cmap='coolwarm')
    plt.show()

plt.figure('silhoutte_score')
plt.plot(x_label, silhoutte_score_list1, label="mds1")
plt.plot(x_label, silhoutte_score_list2, label="mds2")
plt.show()

silhoutte_score_list1 = []
silhoutte_score_list2 = []
x_label = []
for i in range(3, 10):
    t.start("GMM start clustering, num_cluster: %s" % i)
    result1, labels1, silhoutte_score1 = cluster.GMM_cluster(trajectory_list, tmp, i, max_iter=200, metric="euclidean")
    result2, labels2, silhoutte_score2 = cluster.GMM_cluster(trajectory_list, distance_matrix, i, max_iter=200, metric="precomputed")
    t.end("end clustering")
    x_label.append(i)
    silhoutte_score_list1.append(silhoutte_score1)
    silhoutte_score_list2.append(silhoutte_score2)
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels1, cmap='coolwarm')
    plt.show()
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels2, cmap='coolwarm')
    plt.show()

plt.figure('silhoutte_score')
plt.plot(x_label, silhoutte_score_list1, label="mds1")
plt.plot(x_label, silhoutte_score_list2, label="mds2")
plt.show()

silhoutte_score_list1 = []
silhoutte_score_list2 = []
x_label = []
for i in range(3, 15):
    t.start("KMeans start clustering, num_cluster: %s" % i)
    result1, labels1, silhoutte_score1 = cluster.KMeans_cluster(trajectory_list, tmp, i, metric="euclidean")
    result2, labels2, silhoutte_score2 = cluster.KMeans_cluster(trajectory_list, distance_matrix, i,  metric="precomputed")
    t.end("end clustering")
    x_label.append(i)
    silhoutte_score_list1.append(silhoutte_score1)
    silhoutte_score_list2.append(silhoutte_score2)
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels1, cmap='coolwarm')
    plt.show()
    
    plt.figure('mds %s' % i)
    plt.scatter(tmp[:, 0], tmp[:, 1], marker='o', c=labels2, cmap='coolwarm')
    plt.show()

plt.figure('silhoutte_score')
plt.plot(x_label, silhoutte_score_list1, label="mds1")
plt.plot(x_label, silhoutte_score_list2, label="mds2")
plt.show()