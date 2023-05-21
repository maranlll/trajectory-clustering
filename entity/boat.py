from datetime import datetime
import numpy as np
from util.util import geo_to_xy
import json
from util.util import geo_to_v

V_MAX = 30

def s_to_time(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    
def time_to_s(time):
    return datetime.strftime(time, '%Y-%m-%d %H:%M:%S')

def data_to_list(data):
    ret = []
    for d in data:
        tmp = d.tolist()
        tmp[4] = time_to_s(d[4])
        ret.append(tmp)
    return ret

class boat:
    def __init__(self, dais, ais_mmsi, typeId):
        self.dais = dais
        self.ais_mmsi = ais_mmsi
        self.typeId = typeId
        self.data = []
        
    def add_data(self, data):
        lon = data['LON']
        lat = data['LAT']
        cors = data['CORS']
        vel = data['VEL']
        time = data['UDT_TM']
        # x, y = geo_to_xy(lat, lon)
        # tmp = [x, y, cors, vel, s_to_time(time)]
        
        tmp = [lon, lat, cors, vel, s_to_time(time)]
        # tmp = [lon, lat, cors, vel, time]
        self.data.append(tmp)
        
    # def __str__(self):
    #     return json.dumps(dict(self), ensure_ascii=False)
    
    # 按时间排序
    def sort_data(self):
        self.data.sort(key=lambda x: x[4])
        self.data = np.array(self.data)
        
    def extract_data(self):
        new_data = []
        new_data.append(self.data[0])
        last_lon = self.data[0][0]
        last_lat = self.data[0][1]
        last_t = self.data[0][4]
        for i in range(1, len(self.data)):
            # if (abs(self.data[i - 1][0] - self.data[i][0]) < 0.02 and abs(self.data[i - 1][1] - self.data[i][1]) < 0.02):
            #     new_data.append(self.data[i])
            # if ((self.data[i - 1][4] - self.data[i][4]).seconds > 2 * 60 * 60):
                
            if geo_to_v(last_lon, last_lat, self.data[i][0], self.data[i][1], (self.data[i][4] - last_t).seconds) < V_MAX:
                new_data.append(self.data[i])
                last_lon = self.data[i][0]
                last_lat = self.data[i][1]
                last_t = self.data[i][4]
        self.data = np.array(new_data)
        
    def get_points(self, lon_left=118, lon_right=127, lat_bottom=32, width=1000, height=1000):
        result = []
        for d in self.data:
            x, y = geo_to_xy(d[0], d[1], lon_left, lon_right, lat_bottom, width, height)
            result.append([x, y])
        # print(result)
        return np.array(result)
    
    def to_dict(self):
        dict = {}
        dict['dais'] = self.dais
        dict['ais_mmsi'] = self.ais_mmsi
        dict['typeId'] = self.typeId
        dict['data'] = data_to_list(self.data)
        return dict

if (__name__ == "__main__"):
    b = boat(1, 2, 3)
    b.tmp = [1, 2, 3, 4, 5]
    print(b)