import math
from geopy import distance

# map_width = 2000 # in meters
# map_height = 2000  # in meters

# map_lon_left = 119.178395
# map_lon_right = 123.945935
# map_lon_delta = map_lon_right - map_lon_left

# map_lat_bottom = 35
# map_lat_bottom_degree = map_lat_bottom * math.pi / 180


# According to mercator projection it calculates x and y distances in meter relative to top left point
# Taken from https://stackoverflow.com/questions/2103924/mercator-longitude-and-latitude-calculations-to-x-and-y-on-a-cropped-map-of-the/10401734#10401734
def rad(x):
    return x * math.pi / 180

def geo_to_xy(lon, lat, lon_left, lon_right, lat_bottom, width, height):
    map_lon_delta = lon_right - lon_left
    map_lat_bottom_degree = rad(lat_bottom)
    x = (lon - lon_left) * (width / map_lon_delta)
    # print(lon, lon_left, map_lon_delta, x)

    lat = lat * math.pi / 180
    world_map_width = ((width / map_lon_delta) * 360) / (2 * math.pi)
    map_offset_y = (world_map_width / 2 * math.log((1 + math.sin(map_lat_bottom_degree)) / (1 - math.sin(map_lat_bottom_degree))))
    y = (world_map_width / 2 * math.log((1 + math.sin(lat)) / (1 - math.sin(lat)))) - map_offset_y
    # print(lat, map_offset_y, y)
    # x = int(x)
    # y = int(y)
    return x, y

def geo_to_v(lon1, lat1, lon2, lat2, t):
    coord1 = (lat1, lon1)
    coord2 = (lat2, lon2)
    if t == 0:
        return 1000
    d = distance.distance(coord1, coord2).m
    # if d > 3000:
    #     print(d, t)
    v = d / t
    # print(d, v)
    return v