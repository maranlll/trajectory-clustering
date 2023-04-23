import util.point as point
import math
import numpy as np
from util.distance import cal_dVertival_dParallel_dTheta as cal_dvdpdt
from util.distance import cal_distance as cal_dis
from util.point import Point

threshold = 30 * 2 * math.pi / 360 

def shorten_by_angle(trajectory, threshold = threshold, interval = math.inf):
    size = len(trajectory)
    if size <= 2:
        return trajectory
    result = np.array(trajectory[0:2])
    cnt = 0
    for i in range(2, size - 1):
        point1 = point.Point(result[-2])
        point2 = point.Point(result[-1])
        point3 = point.Point(trajectory[i])
        v1 = point.to_numpy(point1, point2)
        v2 = point.to_numpy(point2, point3)
        angel = point.cal_angel(v1, v2)
        if angel > threshold or cnt > interval:
            result = np.vstack((result, trajectory[i]))
            cnt = 0
        else:
            cnt += 1
            
    result = np.vstack((result, trajectory[-1]))
    # print("size before: %s, size after: %s" % (trajectory.shape, result.shape))
    return result
        
def shorten_by_mdl(trajectory, interval = math.inf):
    result = np.array(trajectory[0])
    start = 0
    last = 1
    start_point = Point(trajectory[start])
    last_point = Point(trajectory[last])
    L = cal_dis(start_point, last_point)
    cnt = 0
    for i in range(2, len(trajectory)):
        now_point = Point(trajectory[i])
        LH = np.log2(cal_dis(start_point, now_point))
        L += cal_dis(last_point, now_point)
        dv_cumulate = 0
        dt_cumulate = 0
        for j in range(start + 1, i + 1):
            dv, dp, dt = cal_dvdpdt(Point(trajectory[j - 1]), Point(trajectory[j]), start_point, now_point)
            dv_cumulate += dv
            dt_cumulate += dt

        LDH = 0
        if dv_cumulate != 0:
            LDH += np.log2(dv_cumulate)
        if dt_cumulate != 0:
            LDH += np.log2(dt_cumulate)
        # LDH = np.log2(dv_cumulate) + np.log2(dt_cumulate)

        if LH + LDH > np.log2(L) or cnt > interval:
            result = np.vstack((result, trajectory[last]))
            L = cal_dis(last_point, now_point)
            start = last
            start_point = last_point
            cnt = 0
        else:
            cnt += 1
            
        last = i
        last_point = Point(trajectory[last])
        
    result = np.vstack((result, trajectory[-1]))
    # print("size before: %s, size after: %s" % (trajectory.shape, result.shape))
    return result
    
def segmentation(trajectory_list):
    result = []
    cnt = 0
    for trajectory in trajectory_list:
        for i in range(1, len(trajectory)):
            tmp = dict()
            tmp['points'] = trajectory[i - 1 : i + 1]
            tmp['label'] = cnt
            result.append(tmp)
        cnt += 1
    return result