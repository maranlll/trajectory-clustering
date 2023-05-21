# %%
from util.io import *
from util.similarlity import *
import util.plot as plt
from util.dataPreparation import *
from tqdm import tqdm
from util.timer import Timer

pkl_file = [
    "data/boat/pkl_clean/AIS_TYP_ID_30_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_31_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_32_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_33_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_35_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_36_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_37_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_50_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_51_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_53_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_54_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_55_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_59_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_69_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    "data/boat/pkl_clean/AIS_TYP_ID_89_points.pkl"
]

# save_to = [
    # "data/boat/pkl_clean/AIS_TYP_ID_30_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_31_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_32_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_33_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_35_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_36_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_37_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_50_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_51_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_53_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_54_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_55_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_59_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_69_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_52_points.pkl",
    # "data/boat/pkl_clean/AIS_TYP_ID_89_points.pkl"
# ]

t = Timer()
all = []
cnt = 0

for file in pkl_file:
    print(file)
    t.start("start load data")
    trajectory_list = load(file)
    # trajectory_list = []
    point_num = 0
    for trajectory in tqdm(trajectory_list):
    #     trajectory_list.append(boat.get_points())
        point_num += len(trajectory)
    t.end("end load data")
    all.extend(trajectory_list)
    
    print("point num: %s, trajectory num: %s , average: %s" % (point_num, len(trajectory_list), point_num / len(trajectory_list)))
        
    plt.figure(cnt)
    cnt += 1
    for trajectory in trajectory_list:
        plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.show()
    
plt.figure(cnt)
cnt += 1
for trajectory in all:
    plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.show()

save(all, "data/boat/pkl_clean/all_points.pkl")
