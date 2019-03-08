from hmm import create_connection_dictionary
from hmm import recursive_neighbour_search

import numpy as np

def estimation_accuracy(measurement_edges, estimated_states):
    N = estimated_states.shape[0]
    estimated_edges = np.array([state_space[int(state)]['edge'] for state in estimated_states]) 
    no = 0
    for i in range(N):
        no += int(np.all(np.array(measurement_edges)[i, :] == estimated_edges[i, :]))
    return no/N

def distance_to_true_state(estimated_states, true_states, state_space):
    distances = np.zeros((estimated_states.shape))
    connection_dictionary = create_connection_dictionary(state_space)
    for i, state_id in enumerate(estimated_states):
        distance_dict = recursive_neighbour_search(int(state_id), dict(), connection_dictionary, state_space, 0, 1000)
        distances[i] = distance_dict[true_states[i]]
    return distances