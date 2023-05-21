from scipy.spatial.distance import directed_hausdorff
import math
import numpy as np
from scipy.sparse import lil_matrix
from entity.point import Point, to_numpy
from tqdm import tqdm
import multiprocessing
from util.dtw import dtw, accelerated_dtw

POOL_SIZE = 8

def hausdorff(traj1, traj2):
    return max(directed_hausdorff(traj1, traj2)[0], directed_hausdorff(traj2, traj1)[0])

def dtw_distance(traj1, traj2):
    # return dtw(traj1, traj2, dist=lambda x, y: np.sqrt(np.sum((x - y) ** 2)))[0]
    return accelerated_dtw(traj1, traj2, 'euclidean', 1)[0]

def point_to_vector_distance(p, p1, p2):
    v1 = to_numpy(p1, p)
    v2 = to_numpy(p1, p2)
    if np.linalg.norm(v2) == 0:
        return 0
    return np.linalg.norm(np.cross(v1, v2)) / np.linalg.norm(v2)

def cal_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def cal_dVertival_dParallel_dTheta(p1, p2, p3, p4):
    dv1 = point_to_vector_distance(p1, p3, p4)
    dv2 = point_to_vector_distance(p2, p3, p4)
    if dv1 + dv2 == 0:
        dv = 0
    else:
        dv = (dv1 ** 2 + dv2**2) / (dv1 + dv2)
    dp = min(math.fabs(p3.x - p1.x), math.fabs(p4.x - p2.x))
    dt = math.fabs(dv1 - dv2)
    return dv, dp, dt

def multi_cal(trajectory_list, distance_matrix, f, t, size):
        for i in tqdm(range(f, t)):
            for j in range(i, size):
                distance = hausdorff(trajectory_list[i], trajectory_list[j])
                distance_matrix[i * size + j] = distance
                distance_matrix[j * size + i] = distance

def generate_distance_matrix(trajectory_list, threshold):
    print("start generate distance_matrix")
    size = len(trajectory_list)
    distance_matrix = lil_matrix((size, size))

    for i in tqdm(range(size)):
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
    # distance_matrix = np.empty((size, size))
    
    # for i in tqdm(range(size)):
    #     for j in range(i, size):
    #         distance = hausdorff(trajectory_list[i], trajectory_list[j])
    #         distance_matrix[i, j] = distance
    #         distance_matrix[j, i] = distance
    
    # print("finish generate distance_matrix")
    # return distance_matrix
    
    distance_matrix = multiprocessing.Array('d', size * size)

    p = []
    for i in range(POOL_SIZE):
        tmp = multiprocessing.Process(target=multi_cal, args=(trajectory_list, distance_matrix, int(i * size / POOL_SIZE), int((i + 1) * size / POOL_SIZE), size))
        p.append(tmp)
        tmp.start()
        
    for i in range(len(p)):
        p[i].join()
    
    print("finish generate distance_matrix")
    return np.frombuffer(distance_matrix.get_obj()).reshape((size, size))