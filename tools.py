import numpy as np
import utm

def convert_to_utm(coordinate_array):
    return np.array([utm.from_latlon(coordinate_array[i, 0], coordinate_array[i, 1])[:2]
     for i in range(coordinate_array.shape[0])])

def edges_to_nodes(edge_set):
    node_list = list()
    for edge in edge_set:
        node_list += [edge[0]] + [edge[1]]
    return node_list

def state_sequence_to_node_sequence(state_sequence, state_space):
    node_sequence = list()
    first_state = state_sequence[0]
    last_state = state_sequence[-1]
    for state_id in state_sequence:
        edge = state_space[int(state_id)]['edge']
        if state_id in (first_state, last_state):
            node_sequence.append(edge[0])
            node_sequence.append(edge[1])
        else:
            node_sequence.append(edge[1])

    return node_sequence
        