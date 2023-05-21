# %%
from util.io import *
from util.similarlity import *
import util.plot as plt
from util.dataPreparation import *
from tqdm import tqdm
from util.timer import Timer

pkl_file = [
    # "data/boat/pkl/AIS_TYP_ID_30.pkl",
    # "data/boat/pkl/AIS_TYP_ID_31.pkl",
    # "data/boat/pkl/AIS_TYP_ID_32.pkl",
    # "data/boat/pkl/AIS_TYP_ID_33.pkl",
    # "data/boat/pkl/AIS_TYP_ID_35.pkl",
    # "data/boat/pkl/AIS_TYP_ID_36.pkl",
    # "data/boat/pkl/AIS_TYP_ID_37.pkl",
    # "data/boat/pkl/AIS_TYP_ID_50.pkl",
    # "data/boat/pkl/AIS_TYP_ID_51.pkl",
    "data/boat/pkl/AIS_TYP_ID_52.pkl",
    # "data/boat/pkl/AIS_TYP_ID_53.pkl",
    # "data/boat/pkl/AIS_TYP_ID_54.pkl",
    # "data/boat/pkl/AIS_TYP_ID_55.pkl",
    # "data/boat/pkl/AIS_TYP_ID_59.pkl",
    # "data/boat/pkl/AIS_TYP_ID_69.pkl",
    # "data/boat/pkl/AIS_TYP_ID_52.pkl",
    # "data/boat/pkl/AIS_TYP_ID_89.pkl"
]

save_to = [

    # "data/boat/pkl_clean/AIS_TYP_ID_30_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_31_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_32_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_33_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_35_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_36_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_37_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_50_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_51_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_53_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_54_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_55_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_59_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_69_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_89_points.pkl"
]

t = Timer()
all = []
cnt = 0

for file in pkl_file:
    print(file)
    t.start("start load data")
    boat_list = load(file)
    trajectory_list = []
    point_num = 0
    for boat in tqdm(boat_list):
        trajectory_list.append(boat.get_points())
        point_num += len(boat.get_points())
    t.end("end load data")
        
    plt.figure(1)
    for trajectory in trajectory_list:
        plt.plot(trajectory[:, 0], trajectory[:, 1], ',')
    plt.show()

    # new1 = []  
    t.start("start shorten by angle")
    num_after_shorten1 = 0
    new1 = shorten_by_angle_all_multiprocecss(trajectory_list, 0.01, 50)
    for i in tqdm(range(len(new1))):
        # new1.append(shorten_by_angle(trajectory_list[i], 0.01, 50))
        num_after_shorten1 += len(new1[i])
    t.end("end shorten by angle")

    plt.figure(2)
    for trajectory in new1:
        plt.plot(trajectory[:, 0], trajectory[:, 1], ',')
    plt.show()

    t.start("start shorten by mdl")
    # new2 = [] 
    num_after_shorten2 = 0    
    new2 = shorten_by_mdl_all_multiprocecss(trajectory_list, POOL_SIZE=8, interval=100)
    for i in tqdm(range(len(new2))):
        # new2.append(shorten_by_mdl(trajectory_list[i]))
        num_after_shorten2 += len(new2[i])
    t.end("end shorten by mdl")
    
    save(new2, save_to[cnt])
    cnt = cnt + 1

    plt.figure(3)
    for trajectory in new2:
        plt.plot(trajectory[:, 0], trajectory[:, 1], ',')
    plt.show()   

    print(point_num, num_after_shorten1, num_after_shorten2)
    
    all.extend(new2)
    
save_to(all, "data/boat/pkl_clean/all_points.pkl")