import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

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

from data_wrangling import get_coordinates_of_nodes

from tools import convert_to_utm

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

class MapMatchingVisualization:
    def __init__(self, highways, node_dict, state_space, figsize):
        self.node_dict = node_dict
        self.node_sequences = self.convert_highways_to_nodes(highways, node_dict)
        self.coordinate_arrays = list()
        for node_sequence in self.node_sequences:
            utm_coordinate_array = self.convert_nodes_to_coordinate_array(node_sequence)
            self.coordinate_arrays.append(utm_coordinate_array)
        self.state_space = state_space
        
        self.fig, self.ax = plt.subplots(figsize=figsize);
    
    def convert_nodes_to_coordinate_array(self, nodes):
        coordinate_array = get_coordinates_of_nodes(nodes, self.node_dict)
        return convert_to_utm(coordinate_array)
    
    def convert_highways_to_nodes(self, highways, node_dict):
        list_of_node_sequences = list()
        for highway in highways:
            nodes = highway['data']['nd']
            list_of_node_sequences.append(nodes)
        return list_of_node_sequences
    
    
    
    def plot_road_network(self, color, size, alpha):
        for coord_array in self.coordinate_arrays:
            self.ax.plot(coord_array[:, 0], coord_array[:, 1], lw=size, color=color, alpha=alpha);
        
    def plot_state_sequence(self, state_sequence, color, size, label=False):
        for i, state_id in enumerate(state_sequence):
            state = self.state_space[state_id]
            nodes = list(state['edge'])
            coord_array = self.convert_nodes_to_coordinate_array(nodes)
            self.ax.plot(coord_array[:, 0], coord_array[:, 1], lw=size, color=color)
            if label:
                self.ax.text(np.mean(coord_array[:, 0]) + norm.rvs(0, 10), np.mean(coord_array[:, 1]) + norm.rvs(0,10), str(i));
    
    def plot_bases(self, base_locations, color, size):
        self.ax.scatter(base_locations[:, 0], base_locations[:, 1], color=color, s=size);
        pass
    
    def plot_base_range(self, base_locations, base_max_range):
        for location in base_locations:
            circle = plt.Circle(location, base_max_range, fill=False, color='black')
            self.ax.add_artist(circle);
    
    def shrink_to_fit_state_sequence(self, state_sequence, margin):
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        
        for state_id in state_sequence:
            nodes = list(self.state_space[state_id]['edge'])
            array = self.convert_nodes_to_coordinate_array(nodes)
            if np.min(array[:, 0]) < min_x:
                min_x = np.min(array[:, 0])
            if np.max(array[:, 0]) > max_x:
                max_x = np.max(array[:, 0])
            if np.min(array[:, 1]) < min_y:
                min_y = np.min(array[:, 1])
            if np.max(array[:, 1]) > max_y:
                max_y = np.max(array[:, 1])
        bbox = [min_x, min_y, max_x, max_y]
        self.shrink_plot_to_fit(bbox, margin)
    
    def plot_nodes(self, nodes, color, label=False):
        position = self.convert_nodes_to_coordinate_array(nodes)
        self.ax.plot(position[:, 0], position[:, 1], color=color)
        if label:
            self.ax.text(np.mean(position[:, 0]) + norm.rvs(0, 10), np.mean(position[:, 1]) + norm.rvs(0,10), str(label));
    
    def plot_node_sequence(self, node_sequence, color, label=False):
        for i in range(1, len(node_sequence)):
            if label:
                self.plot_nodes([node_sequence[i-1], node_sequence[i]], color=color, label=i)
            else:
                self.plot_nodes([node_sequence[i-1], node_sequence[i]], color=color)
    
    def plot_coordinate_array(self, array, color, size, signal_measurements=False):
        if isinstance(signal_measurements, np.ndarray):
            max_signal = np.apply_along_axis(np.max, 1, signal_measurements)
            self.ax.scatter(array[:, 0], array[:, 1], c=max_signal, cmap='inferno', s=size)
        else:
            self.ax.scatter(array[:, 0], array[:, 1], color=color, s=size)
    
    def plot_estimation_performance(self, estimated_states, true_states, size, alpha):
        for i, state_id in enumerate(estimated_states):
            correct = true_states[i] == state_id
            if correct:
                color = 'green'
            else:
                color = 'red'
            state = self.state_space[state_id]
            nodes = list(state['edge'])
            coord_array = self.convert_nodes_to_coordinate_array(nodes)
            self.ax.plot(coord_array[:, 0], coord_array[:, 1], color=color, lw=size, alpha=alpha);
            
        
    def shrink_plot_to_fit(self, bbox, margin):
        self.ax.set_xlim(bbox[0] - margin, bbox[2] + margin)
        self.ax.set_ylim(bbox[1] - margin, bbox[3] + margin)
    
    def show_interactive(self):
        return self.fig


    



        
