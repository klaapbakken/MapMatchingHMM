import numpy as np
import utm

from tools import convert_to_utm

def create_node_dict(nodes):
    return {node['data']['id'] : node for node in nodes}

def get_accepted_highways(ways):
    all_highways = [way for way in ways if 'highway' in way['data']['tag'].keys()]
    accepted_highway_types = set(['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service',
                'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
                'living_street', 'pedestrian', 'road', 'escape', 'track'])
    accepted_highways = [highway for highway in all_highways if highway['data']['tag']['highway'] in accepted_highway_types]
    return accepted_highways

def create_highway_dict(highways):
    return {highway['data']['id'] : highway for highway in highways}

def find_shared_nodes(highway_dict):
    node_count = {}
    for highway_id in highway_dict:
        node_list = highway_dict[highway_id]['data']['nd']
        for node in node_list:
            if node not in node_count:
                node_count[node] = (1, [highway_id])
            else:
                node_count[node] = (node_count[node][0] + 1, node_count[node][1] + [highway_id])
    return node_count

def get_required_nodes(highways):
    highway_nodes = [highway['data']['nd'] for highway in highways]
    highway_nodes_array = np.array([node for node_list in highway_nodes for node in node_list])
    required_nodes = np.unique(highway_nodes_array)
    return required_nodes
    
def find_intersections(highway_dict, node_dict):
    node_count = find_shared_nodes(highway_dict)
    intersections = {}
    for node in node_count:
        count = node_count[node][0]
        if count > 1 and node in node_dict:
            intersections[node] = node_count[node][1]
    return intersections

def get_coordinates_of_nodes(node_ids, node_dict):
    node_list = [node_dict[node_id] for node_id in node_ids if node_id in node_dict]
    longitude = np.array([node['data']['lon'] for node in node_list])
    latitude = np.array([node['data']['lat'] for node in node_list])
    return np.array((latitude, longitude)).T

def create_segment(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return lambda x: (y2 - y1)/(x2 - x1)*(x - x1) + y1

def create_segment_list(node_ids, node_dict):
    coordinate_array = get_coordinates_of_nodes(node_ids, node_dict)
    utm_coordinate_array = convert_to_utm(coordinate_array)
    return [create_segment(utm_coordinate_array[i, :], utm_coordinate_array[i+1, :]) 
    for i in range(utm_coordinate_array.shape[0]-1)], utm_coordinate_array

def create_state_space_representations(highways, node_dict):
    id_tag = 0
    state_space = list()
    for highway in highways:
        for i in range(1, len(highway['data']['nd'])):
            node_a = node_dict[highway['data']['nd'][i-1]]
            node_b = node_dict[highway['data']['nd'][i]]

            latlon_a = (node_a['data']['lat'], node_a['data']['lon'])
            latlon_b = (node_b['data']['lat'], node_b['data']['lon'])

            coords_a = convert_to_utm(np.array(latlon_a).reshape(1, 2))
            coords_b = convert_to_utm(np.array(latlon_b).reshape(1, 2))

            if not np.all(coords_a == coords_b):

                state = {'id' : id_tag,
                'edge' : (node_a['data']['id'], node_b['data']['id']),
                'edge_set' : set((node_a['data']['id'], node_b['data']['id'])),
                'function' : create_segment(coords_a.reshape((2,)), coords_b.reshape((2,))),
                'domain' : (coords_a[0, 0], coords_b[0, 0])
                }
            
                state_space.append(state)
                id_tag += 1
    
    return state_space