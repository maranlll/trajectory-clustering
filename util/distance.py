from scipy.spatial.distance import directed_hausdorff
import math
import numpy as np
from scipy.sparse import lil_matrix

def hausdorff(traj1, traj2):
    return max(directed_hausdorff(traj1, traj2)[0], directed_hausdorff(traj2, traj1)[0])

def generate_distance_matrix(trajectory_list, threshold):
    print("start generate distance_matrix")
    size = len(trajectory_list)
    distance_matrix = lil_matrix((size, size))

    for i in range(size):
        for j in range(i, size):
            distance = hausdorff(trajectory_list[i], trajectory_list[j])
            if (distance < threshold):
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    print("finish generate distance_matrix")
    return distance_matrix

def generate_dense_distance_matrix(trajectory_list):
    print("start generate dense distance_matrix")
    size = len(trajectory_list)
    distance_matrix = np.empty((size, size))

    for i in range(size):
        for j in range(i, size):
            distance = hausdorff(trajectory_list[i], trajectory_list[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    print("finish generate distance_matrix")
    return distance_matrix