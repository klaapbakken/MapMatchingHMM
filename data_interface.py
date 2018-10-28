import psycopg2
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
from osmapi import OsmApi
import json
import os

from tools import convert_to_utm


def load_coords_from_json(json_file):
    with open(json_file) as f:
        track = json.load(f)
    longitude = list(filter(lambda x: x != 0.0, [track['points'][i]['lon'] for i in range(len(track['points']))]))
    latitude = list(filter(lambda x: x != 0.0, [track['points'][i]['lat'] for i in range(len(track['points']))]))
    coordinate_array = np.array((latitude, longitude)).T
    utm_coordinate_array = convert_to_utm(coordinate_array)
    return utm_coordinate_array, coordinate_array

def load_coords_from_folder(json_folder):
    utm_coordinate_array = np.empty((0,2))
    coordinate_array = np.empty((0,2))
    json_files = os.listdir(json_folder)
    for file in json_files:
        abs_path = os.path.join(json_folder, file)
        utm_coordinates, coordinates = load_coords_from_json(abs_path)
        utm_coordinate_array = np.concatenate((utm_coordinate_array, utm_coordinates))
        coordinate_array = np.concatenate((coordinate_array, coordinates))
    return utm_coordinate_array, coordinate_array

def node_row_to_dict(row):
    e = list(filter(lambda x: any([i.isalnum() for i in x]), row['tags'].split('"')))
    lonlat = mapping(row['geom'])['coordinates']
    node_dict = {'data' : {'id' : row['id'],
                'lat' : lonlat[1],
                'lon' : lonlat[0],
                'tag' : {e[2*i] : e[2*i + 1] for i in range(len(e)//2)}
                },
                'type' : 'node'}
    return node_dict

def ways_row_to_dict(row):
    e = list(filter(lambda x: any([i.isalnum() for i in x]), row['tags'].split('"')))
    ways_dict = {'data' : {'id' : row['id'],
                'nd' : row['nodes'],
                'tag' : {e[2*i] : e[2*i + 1] for i in range(len(e)//2)}
                },
                'type' : 'way'}
    return ways_dict

def query_nodes_postgis_db(node_ids, password):
    initial_query = "select * from nodes where id = " + str(node_ids[0])
    base_query = " union select * from nodes where id = "
    sql_query = initial_query
    for node_id in node_ids[1:]:
        sql_query += base_query + str(node_id)
    sql_query += ';'
    con = psycopg2.connect(database="geodatabase", user="postgres", password=password,
        host="localhost")
    node_df = gpd.GeoDataFrame.from_postgis(sql_query, con, geom_col='geom')
    nodes = []
    for i in range(len(node_df)):
        row = node_df.iloc[i, :]
        nodes.append(node_row_to_dict(row))
    return nodes

def query_osm_api(bbox):
    my_api = OsmApi()
    city = my_api.Map(bbox[0], bbox[1], bbox[2], bbox[3])
    nodes = [element for element in city if element['type'] == 'node']
    ways = [element for element in city if element['type'] == 'way']
    return nodes, ways

def query_ways_postgis_db(bbox, password):
    con = psycopg2.connect(database="geodatabase", user="postgres", password=password,
        host="localhost")
    ways_sql = "select * from ways where ways.linestring && ST_MakeEnvelope({0}, {1}, {2}, {3});".format(bbox[0], bbox[1], bbox[2], bbox[3])
    ways_df = gpd.GeoDataFrame.from_postgis(ways_sql, con, geom_col='linestring' )
    ways = []
    for j in range(len(ways_df)):
        row = ways_df.iloc[j, :]
        ways.append(ways_row_to_dict(row))
    return ways