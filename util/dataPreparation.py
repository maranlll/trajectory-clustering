import entity.point as point
import math
import numpy as np
from util.similarlity import cal_dVertival_dParallel_dTheta as cal_dvdpdt
from util.similarlity import cal_distance as cal_dis
from entity.point import Point
import multiprocessing
from tqdm import tqdm

threshold = 1 * math.pi / 180

def shorten_by_angle(trajectory, threshold, interval):
    size = len(trajectory)
    if size <= 2:
        return trajectory
    # result = np.array(trajectory[0:2])
    result = []
    result.append(trajectory[0])
    result.append(trajectory[1])
    cnt = 0
    for i in range(2, size - 1):
        point1 = point.Point(result[-2])
        point2 = point.Point(result[-1])
        point3 = point.Point(trajectory[i])
        v1 = point.to_numpy(point1, point2)
        v2 = point.to_numpy(point2, point3)
        angel = point.cal_angel(v1, v2)
        # print("angel: %s, cnt: %s" % (angel, cnt))
        if angel > threshold or cnt > interval:
            # print("angel: %s, cnt: %s" % (angel, cnt))
            # result = np.vstack((result, trajectory[i]))
            result.append(trajectory[i])
            cnt = 0
        else:
            cnt += 1
            
    # result = np.vstack((result, trajectory[-1]))
    result.append(trajectory[-1])
    # print("size before: %s, size after: %s" % (trajectory.shape, result.shape))
    # return result
    return np.array(result)
        
def shorten_by_mdl(trajectory, interval = math.inf):
    # print(trajectory)
    if len(trajectory) <= 2:
        return trajectory
    # result = np.array(trajectory[0])
    result = []
    result.append(trajectory[0])
    start = 0
    last = 1
    start_point = Point(trajectory[start])
    last_point = Point(trajectory[last])
    L = 1e-6
    L += cal_dis(start_point, last_point)
    cnt = 0
    for i in range(2, len(trajectory)):
        now_point = Point(trajectory[i])
        LH = np.log2(cal_dis(start_point, now_point) + 1e-5)
        L += cal_dis(last_point, now_point)
        dv_cumulate = 1e-6
        dt_cumulate = 1e-6
        for j in range(start + 1, i + 1):
            dv, dp, dt = cal_dvdpdt(Point(trajectory[j - 1]), Point(trajectory[j]), start_point, now_point)
            dv_cumulate += dv
            dt_cumulate += dt

        LDH = np.log2(dv_cumulate) + np.log2(dt_cumulate)
        
        # print(cal_dis(start_point, now_point), L, dv_cumulate, dt_cumulate, LH + LDH, np.log2(L))

        if LH + LDH > np.log2(L) or cnt > interval:
            # print(cal_dis(start_point, now_point), L, dv_cumulate, dt_cumulate, LH + LDH, np.log2(L), LH + LDH > np.log2(L), cnt, interval)
            # result = np.vstack((result, trajectory[last]))
            result.append(trajectory[last])
            L = cal_dis(last_point, now_point) + 1e-6
            start = last
            start_point = last_point
            cnt = 0
        else:
            cnt += 1
            
        last = i
        last_point = Point(trajectory[last])
        
    # result = np.vstack((result, trajectory[-1]))
    result.append(trajectory[-1])
    # print("size before: %s, size after: %s" % (trajectory.shape, result.shape))
    # return result
    return np.array(result)
  
def shorten_by_angle_all(trajectory_list, start, end, threshold=threshold, interval=math.inf):
    l = []
    for i in tqdm(range(start, end)):
        l.append(shorten_by_angle(trajectory_list[i], threshold, interval))
    return l
    
def shorten_by_mdl_all(trajectory_list, start, end, interval=math.inf):
    l = []
    for i in tqdm(range(start, end)):
        l.append(shorten_by_mdl(trajectory_list[i], interval))
    return l

def shorten_by_angle_all_multiprocecss(trajectory_list, threshold=threshold, interval=math.inf, POOL_SIZE=8):
    pool = multiprocessing.Pool(processes=POOL_SIZE)
    size = len(trajectory_list)

    p = []
    for i in range(POOL_SIZE):
        p.append(pool.apply_async(shorten_by_angle_all, args=(trajectory_list, int(i * size / POOL_SIZE), int((i + 1) * size / POOL_SIZE), threshold, interval,)))
    pool.close()
    pool.join()
    
    ret = []
    for i in p:
        ret.extend(i.get())
        
    # print(len(trajectory_list), len(ret))
    return ret

def shorten_by_mdl_all_multiprocecss(trajectory_list, interval=math.inf, POOL_SIZE=8): 
    pool = multiprocessing.Pool(processes=POOL_SIZE)
    size = len(trajectory_list)

    p = []
    for i in range(POOL_SIZE):
        p.append(pool.apply_async(shorten_by_mdl_all, args=(trajectory_list, int(i * size / POOL_SIZE), int((i + 1) * size / POOL_SIZE), interval,)))
    pool.close()
    pool.join()
    
    ret = []
    for i in p:
        ret.extend(i.get())
        
    # print(len(trajectory_list), len(ret))
    return ret
    
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