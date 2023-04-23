import numpy as np

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
    if v1_norm == 0 or v2_norm == 0:
        return 0
    return np.arccos(np.dot(v1 / v1_norm, v2 / v2_norm))

class Segment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.v = to_numpy(p1, p2)
        self.v_norm = np.linalg.norm(self.v)
        self.v_unit = self.v / self.v_norm