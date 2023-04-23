# %%
import util.plot as plt
import util.io as io
import util.distance as dis
import util.clustering as cluster
from util.timer import Timer
import numpy as np
import util.dataPreparation as data
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

eps = 22
min_cluster_size = 10
min_samples = 10
threshold = 50
iter_max = 2000

filename = 'data/1/cross.mat'
trajectory_list = io.read_from_mat(filename)
t = Timer()

print("start data preparation")
for i in tqdm(range(len(trajectory_list))):
    trajectory_list[i] = data.shorten_by_mdl(trajectory_list[i])
print("end data preparation")
        
# %% 
print("start segementation")     
trajectory_segment = data.segmentation(trajectory_list)

trajectory_segment_list = []
for semgent in trajectory_segment:
    trajectory_segment_list.append(semgent['points'])

# print(trajectory_segment_list)
trajectory_list = trajectory_segment_list
print("end segementation")     

# %%
# t.start()
# print("trajectory size: %s" % len(trajectory_list))
# # distance_matrix = dis.generate_distance_matrix(trajectory_list, threshold)
# distance_matrix = dis.generate_dense_distance_matrix(trajectory_list)
# t.end("generate distance_matrix")
# 
# io.save(distance_matrix, "data/1/cross_segment.pkl")

# distance_matrix = io.load_distance_matrix("data/cross.npz")
distance_matrix = io.load("data/1/cross_segment.pkl")
# print(distance_matrix)

t.start()
num_cluster, result, noise = cluster.HDBSCAN_cluster(trajectory_list, distance_matrix, min_cluster_size, min_samples, "precomputed")
# num_cluster = 8
# result= cluster.GMM_cluster(trajectory_list, distance_matrix, num_cluster, max_iter=iter_max)
t.end("clustering")

plt.figure(1)
# plt.subplot(121)
for trajectory in trajectory_list:
    plt.plot(trajectory[:, 0], trajectory[:, 1])

plt.subplot(122)

color_map = plt.get_rand_color_map(num_cluster)

for label in range(num_cluster):
    for trajectory in result[label]:
        plt.plot_trajectory(trajectory, color_map(label), 0.3)
        
for trajectory in noise:
    plt.plot_trajectory(trajectory, "black", 0.3)
        
# for trajectory in core:
#     plt.plot_trajectory(trajectory, "black", 1)

plt.show()