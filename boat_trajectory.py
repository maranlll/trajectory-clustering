# %%
import util.io as io
import util.plot as plt
from tqdm import tqdm
import util.similarlity as dis
import numpy as np

import util.dataPreparation as dataPre

# data = io.load("data/boat/boat_data_53.pkl")

# print(len(data))

# max_lon = -1000
# min_lon = 1000
# lat_bottom = 1000
# width = 1000
# height = 3000

# %%
def show_trajectory(data):
    max_lon = -1000
    min_lon = 1000
    lat_bottom = 1000
    width = 1000
    height = 3000
    # lat_array = np.array([])
    # lon_array = np.array([])
    for d in tqdm(data):
        # print(d.data)
        lat_bottom = min(lat_bottom, d.data[:, 1].min())
        max_lon = max(max_lon, d.data[:, 0].max())
        min_lon = min(min_lon, d.data[:, 0].min())
        # lat_array = np.append(lat_array, d.data[:, 1])
        # lon_array = np.append(lon_array, d.data[:, 0])
        
    # lat_array.sort()
    # lon_array.sort()    
    print(max_lon, min_lon, lat_bottom)
    # print(lat_array[:100])
    # print(lon_array[:100])
    # print(lat_array[-100:])
    # print(lon_array[-100:])
    
    trajectory_list = []

    for d in tqdm(data):
        # if (len(d.data) < 100):
        #     continue
        trajectory_list.append(d.get_points(min_lon, max_lon, lat_bottom, width, height))
        
    plt.figure(1)

    for t in trajectory_list:
        # plt.plot(t[:, 0], t[:, 1], ',')
        plt.plot(t[:, 0], t[:, 1])

    plt.show()

# %%
def short_data(trajectory_list):
    print("start data preparation")
    for i in tqdm(range(len(trajectory_list))):
        trajectory_list[i] = dataPre.shorten_by_mdl(trajectory_list[i])
    print("end data preparation")

    print("start generate distance_matrix")
    print("trajectory size: %s" % len(trajectory_list))
    # distance_matrix = dis.generate_distance_matrix(trajectory_list, threshold)
    distance_matrix = dis.generate_dense_distance_matrix(trajectory_list)
    print("end generate distance_matrix")

    # io.save(distance_matrix, "data/boat/distance_52.pkl")
    # io.save(trajectory_list, "data/boat/trajectory_52.pkl")

#%%
if __name__=="__main__":
    file_list = [
                # "data/boat/AIS_TYP_ID_30.json",
                #  "data/boat/AIS_TYP_ID_31.json",
                #  "data/boat/AIS_TYP_ID_32.json",
                #  "data/boat/AIS_TYP_ID_33.json",
                #  "data/boat/AIS_TYP_ID_35.json",
                #  "data/boat/AIS_TYP_ID_36.json",
                #  "data/boat/AIS_TYP_ID_37.json",
                #  "data/boat/AIS_TYP_ID_50.json",
                #  "data/boat/AIS_TYP_ID_51.json",
                #  "data/boat/AIS_TYP_ID_52.json",
                 "data/boat/AIS_TYP_ID_53.json",
                #  "data/boat/AIS_TYP_ID_54.json",
                #  "data/boat/AIS_TYP_ID_55.json",
                #  "data/boat/AIS_TYP_ID_59.json",
                #  "data/boat/AIS_TYP_ID_69.json",
                #  "data/boat/AIS_TYP_ID_89.json"
                 ]

    data = io.load(file_list[0])
    show_trajectory(data)