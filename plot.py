# %%
import util.io as io
import util.plot as plt
import util.clustering as cluster
from util.timer import Timer

t = Timer()
threshold = 50
min_cluster_size = 10
min_samples = 10

distance_matrix = io.load("data/boat/distance_53.pkl")
trajectory_list = io.load("data/boat/trajectory_53.pkl")

# %%
t.start()
num_cluster, result, noise = cluster.HDBSCAN_cluster(trajectory_list, distance_matrix, min_cluster_size, min_samples, "precomputed")
# num_cluster = 8
# result= cluster.GMM_cluster(trajectory_list, distance_matrix, num_cluster, max_iter=iter_max)
t.end("clustering")

# %% 
plt.figure(1)
plt.subplot(131)
for trajectory in trajectory_list:
    plt.plot(trajectory[:, 0], trajectory[:, 1])

plt.subplot(132)

color_map = plt.get_rand_color_map(num_cluster)

for label in range(num_cluster):
    for trajectory in result[label]:
        plt.plot_trajectory(trajectory, color_map(label), 0.7)
    
plt.subplot(133)    
for trajectory in noise:
    plt.plot_trajectory(trajectory, "black", 0.3)

plt.show()