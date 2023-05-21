# %%
from boat_trajectory import *
from boat_data import *
from util.io import *

file_list = [
             "data/boat/raw/AIS_TYP_ID_30.json",
             "data/boat/raw/AIS_TYP_ID_31.json",
             "data/boat/raw/AIS_TYP_ID_32.json",
             "data/boat/raw/AIS_TYP_ID_33.json",
             "data/boat/raw/AIS_TYP_ID_35.json",
             "data/boat/raw/AIS_TYP_ID_36.json",
             "data/boat/raw/AIS_TYP_ID_37.json",
             "data/boat/raw/AIS_TYP_ID_50.json",
             "data/boat/raw/AIS_TYP_ID_51.json",
             "data/boat/raw/AIS_TYP_ID_52.json",
             "data/boat/raw/AIS_TYP_ID_53.json",
             "data/boat/raw/AIS_TYP_ID_54.json",
             "data/boat/raw/AIS_TYP_ID_55.json",
             "data/boat/raw/AIS_TYP_ID_59.json",
             "data/boat/raw/AIS_TYP_ID_69.json",
             "data/boat/raw/AIS_TYP_ID_89.json"
]

save_to_json = [
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_30.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_31.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_32.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_33.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_35.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_36.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_37.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_50.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_51.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_52.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_53.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_54.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_55.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_59.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_69.json",
    "data/boat/lon_lat_cors_v_t/AIS_TYP_ID_89.json"
]

save_to_pkl = [
    "data/boat/pkl/AIS_TYP_ID_30.pkl",
    "data/boat/pkl/AIS_TYP_ID_31.pkl",
    "data/boat/pkl/AIS_TYP_ID_32.pkl",
    "data/boat/pkl/AIS_TYP_ID_33.pkl",
    "data/boat/pkl/AIS_TYP_ID_35.pkl",
    "data/boat/pkl/AIS_TYP_ID_36.pkl",
    "data/boat/pkl/AIS_TYP_ID_37.pkl",
    "data/boat/pkl/AIS_TYP_ID_50.pkl",
    "data/boat/pkl/AIS_TYP_ID_51.pkl",
    "data/boat/pkl/AIS_TYP_ID_52.pkl",
    "data/boat/pkl/AIS_TYP_ID_53.pkl",
    "data/boat/pkl/AIS_TYP_ID_54.pkl",
    "data/boat/pkl/AIS_TYP_ID_55.pkl",
    "data/boat/pkl/AIS_TYP_ID_59.pkl",
    "data/boat/pkl/AIS_TYP_ID_69.pkl",
    "data/boat/pkl/AIS_TYP_ID_89.pkl"
]

all_boat = []
cnt = 0
for file in file_list:
    tmp = get_boat(file)
    print(len(tmp))
    show_trajectory(tmp)
    all_boat += tmp
    
    json_data = []
    print("save to: ", save_to_json[cnt])
    for boat in tqdm(tmp):
        d = boat.to_dict()
        json_data.append(d)
    print(len(json_data))    
    # save_json(json_data, save_to_json[cnt])
    # save(tmp, save_to_pkl[cnt])
    cnt += 1
    
show_trajectory(all_boat)
# save(all_boat, "data/boat/pkl/all_boat.pkl")