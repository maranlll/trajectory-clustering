import numpy as np
import scipy.io
import pickle
import scipy.sparse

def read_from_mat(fileName):
    print("start read data")
    trajectory_data = scipy.io.loadmat(fileName)['tracks']
    trajectory_list = []
    for data_instance in trajectory_data:
        trajectory_list.append(np.vstack(data_instance[0]).T)
    print("trajectory size: %s" % len(trajectory_list))
    return trajectory_list

def save(data, fileName):
    f = open(fileName, "wb")
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def load(fileName):
    f = open(fileName, "rb")
    d = pickle.load(f)
    f.close()
    return d

def save_distance_matrix(x, filename):
    x = scipy.sparse.csr_matrix(x)
    scipy.sparse.save_npz(filename, x)

def load_distance_matrix(filename):
    d = scipy.sparse.load_npz(filename)
    d = scipy.sparse.lil_matrix(d)
    return d