import matplotlib.pyplot as plt
import numpy as np

from data_interface import query_ways_postgis_db
from data_interface import query_nodes_postgis_db

from data_wrangling import get_accepted_highways
from data_wrangling import create_node_dict
from data_wrangling import create_highway_dict
from data_wrangling import get_required_nodes
from data_wrangling import get_coordinates_of_nodes

from tools import convert_to_utm


def plot_highway(highway, node_dict, color, alpha=1):
    node_ids = highway['data']['nd']
    coordinate_array = get_coordinates_of_nodes(node_ids, node_dict)
    utm_coordinate_array = convert_to_utm(coordinate_array)
    plt.plot(utm_coordinate_array[:, 0], utm_coordinate_array[:, 1], c=color, alpha=alpha)

def plot_nodes(node_ids, node_dict, color, limit_axis=True, scatter=False, alpha=1, linestyle='-'):
    coordinate_array = get_coordinates_of_nodes(node_ids, node_dict)
    utm_coordinate_array = convert_to_utm(coordinate_array)
    if scatter:
        plt.scatter(utm_coordinate_array[:, 0], utm_coordinate_array[:, 1], c=color, alpha=alpha)
    else:
        plt.plot(utm_coordinate_array[:, 0], utm_coordinate_array[:, 1], c=color, alpha=alpha, linestyle=linestyle)
    if limit_axis:
        ax = plt.gca()
        ax.set_xlim([np.min(utm_coordinate_array[:, 0])-100, np.max(utm_coordinate_array[:, 0])+100])
        ax.set_ylim([np.min(utm_coordinate_array[:, 1])-100, np.max(utm_coordinate_array[:, 1])+100])
        ax.set_aspect(1.0)

def plot_road_network(highways, node_dict, intersections):
    for highway in highways:
        plot_highway(highway, node_dict, 'b', alpha=0.2)
    plot_nodes(list(intersections.keys()), node_dict, 'y', scatter=True)
    plt.show()

def plot_route(highways, node_dict, route, measurements=None):
    for highway in highways:
        plot_highway(highway, node_dict, 'b', alpha=0.2)
    plot_nodes(route, node_dict, 'y', scatter=False)
    if measurements is not None:
        plt.scatter(measurements[:, 0], measurements[:, 1], s=50, c='r')
    plt.show()

def plot_edge(edge, node_dict, color, alpha=1, linestyle='-'):
    node_ids = list(edge)
    coordinate_array = get_coordinates_of_nodes(node_ids, node_dict)
    utm_coordinate_array = convert_to_utm(coordinate_array)
    plt.plot(utm_coordinate_array[:, 0], utm_coordinate_array[:, 1], c=color, alpha=alpha, linestyle=linestyle)

def plot_results(state_space, node_dict, measurements, true_edges, estimated_states):
    for state in state_space:
        plot_edge(state['edge'], node_dict, color='b', alpha=0.2)
    plt.scatter(measurements[:, 0], measurements[:, 1], s=5, color='m')
    for edge in true_edges:
        plot_edge(edge, node_dict, color='g', alpha=0.8)
    for state in estimated_states:
        plot_edge(state_space[int(state)]['edge'], node_dict, color='y', linestyle=':')
    ax = plt.gca()
    ax.set_xlim([np.min(measurements[:, 0])-100, np.max(measurements[:, 0])+100])
    ax.set_ylim([np.min(measurements[:, 1])-100, np.max(measurements[:, 1])+100])
    plt.show()


    



        
