import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Point:
    def __init__(self, p):
        self.x = p[0]
        self.y = p[1]
        
    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __minus__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
def to_numpy(point1, point2):
    return np.array([point2.x - point1.x, point2.y - point1.y])

def cal_angel(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm * v2_norm == 0 :
        return 0
    tmp = np.dot(v1 / v1_norm, v2 / v2_norm)
    
    if (1 - abs(tmp) < 1e-6):
        return 0
    return np.arccos(tmp)
    # return np.arccos(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0])

class Segment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.v = to_numpy(p1, p2)
        self.v_norm = np.linalg.norm(self.v)
        self.v_unit = self.v / self.v_norm
        
if __name__ == '__main__':
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])