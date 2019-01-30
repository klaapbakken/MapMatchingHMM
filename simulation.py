import numpy as np
from scipy.stats import beta, norm, bernoulli

from data_wrangling import create_segment_list

from tools import edges_to_states

def simulate_route(highway_dict, starting_node, starting_highway, intersections, n):
    current_highway_id = starting_highway
    current_node_id = starting_node
    move = True
    steps = 0
    route = []
    direction = 1

    while move is True:
        route.append(current_node_id)
        if current_node_id in intersections:
            new_highway_candidates = intersections[current_node_id]
            new_highway_id = np.random.choice(new_highway_candidates)
            new_highway_nodes = highway_dict[new_highway_id]['data']['nd']
            position_on_new_highway = np.where(current_node_id == np.array(new_highway_nodes))[0][0]
            if position_on_new_highway == len(new_highway_nodes) - 1:
                direction = -1
            elif position_on_new_highway == 0:
                direction = 1
            current_node_id = new_highway_nodes[position_on_new_highway + direction]
            current_highway_id = new_highway_id
        else:
            current_highway_nodes = highway_dict[current_highway_id]['data']['nd']
            position_on_current_highway = np.where(current_node_id == np.array(current_highway_nodes))[0][0]
            if position_on_current_highway == len(current_highway_nodes) - 1:
                direction = -1
            elif position_on_current_highway == 0:
                direction = 1
            current_node_id = current_highway_nodes[position_on_current_highway + direction]
        steps += 1
        if steps > n:
            move = False   
    return route

def simulate_gps_signals(route, node_dict, variance, polling_rate, max_speed):
    measurements = []
    measurement_edges = []
    segments, coordinate_array = create_segment_list(route, node_dict)
    measurements.append((coordinate_array[0, 0] + norm.rvs(0, variance), coordinate_array[0, 1] + norm.rvs(0, variance)))
    measurement_edges.append((route[0], route[1]))
    remaining_space = 0
    offset = 0
    for i in range(len(route) - 1):
        edge = (route[i], route[i + 1])
        #Segment
        s_i = segments[i]
        #Speed limit
        l_i = max_speed[i]
        #Length of segment
        d_i = np.linalg.norm(coordinate_array[i+1, :] - coordinate_array[i, :].T)
        #Total time when driving at speed limit
        t_i = d_i/l_i
        #Number of measurements
        m_i = t_i*polling_rate
        #Distance between each measurement
        dpm = d_i/m_i
        #Slope
        a = (coordinate_array[i + 1, 1] - coordinate_array[i, 1])/(np.abs(coordinate_array[i + 1, 0] - coordinate_array[i, 0]))
        #Angle
        theta = np.arctan(a)
        #Direction of movement on x - axis
        direction = np.sign(coordinate_array[i+1, 0] - coordinate_array[i, 0])
        #Available space
        available_space = d_i
        required_space = dpm - remaining_space
        #Counter
        #While points can be added on current segment
        while available_space >= required_space:
            #Length of movement along x - axis
            delta_x = np.cos(theta)*(required_space)
            offset += delta_x
            #Measurements
            x = coordinate_array[i, 0]
            measurements.append((x + direction*offset + norm.rvs(0, variance), s_i(x + direction*offset) + norm.rvs(0, variance)))
            measurement_edges.append(edge)
            #New distance on segment has been covered
            available_space -= required_space
            #Residual distance is zero after adding it once on one segment
            required_space = dpm
            remaining_space = 0
            #Iterating counter
        #Residual distance is updated.
        remaining_space += available_space
        offset = 0
    return np.array(measurements), np.array(measurement_edges)

def simulate_gps_observations(route, node_dict, variance, polling_rate, max_speed):
    true_positions = []
    measurements = []
    measurement_edges = []
    segments, coordinate_array = create_segment_list(route, node_dict)
    measurements.append((coordinate_array[0, 0] + norm.rvs(0, variance), coordinate_array[0, 1] + norm.rvs(0, variance)))
    true_positions.append((coordinate_array[0, 0], coordinate_array[0, 1]))
    measurement_edges.append((route[0], route[1]))
    remaining_space = 0
    offset = 0
    for i in range(len(route) - 1):
        edge = (route[i], route[i + 1])
        #Segment
        s_i = segments[i]
        #Speed limit
        l_i = max_speed[i]
        #Length of segment
        d_i = np.linalg.norm(coordinate_array[i+1, :] - coordinate_array[i, :].T)
        #Total time when driving at speed limit
        t_i = d_i/l_i
        #Number of measurements
        m_i = t_i*polling_rate
        #Distance between each measurement
        dpm = d_i/m_i
        #Slope
        a = (coordinate_array[i + 1, 1] - coordinate_array[i, 1])/(np.abs(coordinate_array[i + 1, 0] - coordinate_array[i, 0]))
        #Angle
        theta = np.arctan(a)
        #Direction of movement on x - axis
        direction = np.sign(coordinate_array[i+1, 0] - coordinate_array[i, 0])
        #Available space
        available_space = d_i
        required_space = dpm - remaining_space
        #Counter
        #While points can be added on current segment
        while available_space >= required_space:
            #Length of movement along x - axis
            delta_x = np.cos(theta)*(required_space)
            offset += delta_x
            #Measurements
            x = coordinate_array[i, 0]
            measurements.append((x + direction*offset + norm.rvs(0, variance), s_i(x + direction*offset) + norm.rvs(0, variance)))
            true_positions.append((x + direction*offset, s_i(x + direction*offset)))
            measurement_edges.append(edge)
            #New distance on segment has been covered
            available_space -= required_space
            #Residual distance is zero after adding it once on one segment
            required_space = dpm
            remaining_space = 0
            #Iterating counter
        #Residual distance is updated.
        remaining_space += available_space
        offset = 0
    return np.array(measurements), np.array(measurement_edges), np.array(true_positions)

from scipy.stats import beta, bernoulli

def generate_signal_strength(measurement_locations, base_locations, base_max_ranges):
    measurement_observations = np.zeros((measurement_locations.shape[0], base_locations.shape[0]))
    for measurement_id, position in enumerate(measurement_locations):
        signal_strengths = list()
        for base_index, base_position in enumerate(base_locations):
            distance = np.linalg.norm(position - base_position)
            signal_received = bernoulli.rvs(max(0, 1 - distance/(base_max_ranges[base_index])))
            signal_strength = 0
            if signal_received:
                signal_strength = beta.rvs(2, 5*distance/base_max_ranges[base_index])
            signal_strengths.append(signal_strength*signal_received)
        measurement_observations[measurement_id, :] = np.array(signal_strengths)
    return measurement_observations 

def simulate_observations(route, node_dict, gps_variance, gps_polling_rate, route_speed_limits, base_locations, base_max_ranges, state_space):
    gps_measurements, measurement_edges, true_measurement_positions = simulate_gps_observations(route, node_dict, gps_variance, gps_polling_rate, route_speed_limits)
    measurement_states = edges_to_states(measurement_edges, state_space)
    signal_measurements = generate_signal_strength(true_measurement_positions, base_locations, base_max_ranges)
    return gps_measurements, signal_measurements, measurement_states