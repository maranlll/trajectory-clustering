# %%
import util.io as io
from entity.boat import boat as Boat
from tqdm import tqdm
import numpy as np

V_MAX = 16
LEN_MIN = 4

# return boat_list
def get_boat(file):
    print('start ' + file)
    data = io.load_json(file)

    boat_list = []
    boat_map = dict()

    for d in tqdm(data):
        dais = d['DAIS']
        if (d['LON'] > 127 or d['LON'] < 118 or d['LAT'] > 37 or d['LAT'] < 32):
            continue
        if dais not in boat_map:
            tmp = Boat(d['DAIS'], d['AIS_MMSI'], d['SHP_TYP_ID'])
            boat_list.append(tmp)
            boat_map[dais] = tmp
        boat = boat_map[dais]
        boat.add_data(d)
      
    ret = []  
    for boat in tqdm(boat_list):
        if (len(boat.data) > LEN_MIN):
            boat.sort_data()
            boat.extract_data()
            ret.append(boat)
    
    return ret

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
                #  "data/boat/AIS_TYP_ID_53.json",
                #  "data/boat/AIS_TYP_ID_54.json",
                #  "data/boat/AIS_TYP_ID_55.json",
                 "data/boat/AIS_TYP_ID_59.json",
                #  "data/boat/AIS_TYP_ID_69.json",
                #  "data/boat/AIS_TYP_ID_89.json"
                 ]

    data = []
    for file in file_list:
        data += get_boat(file)
    
    io.save(data, "data/boat/boat_data_53.pkl")
    # io.save_json(data, "data/boat/boat_data_all.json")