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

def get_accuracy_of_estimate(measurement_edges, estimated_states, state_space):
    N = estimated_states.shape[0]
    estimated_edges = np.array([state_space[int(state)]['edge'] for state in estimated_states]) 
    no = 0
    for i in range(N):
        no += int(np.all(np.array(measurement_edges)[i, :] == estimated_edges[i, :]))
    return no/N

def edge_to_state(edge, state_space):
    unordered_edge = set(edge)
    search = True
    for state in state_space:
        if state['edge_set'] == unordered_edge:
            return state['id']
    return None

def edges_to_states(edge_array, state_space):
    states = list()
    for row in range(edge_array.shape[0]):
        states.append(edge_to_state(edge_array[row, :], state_space))
    return states