import numpy as np

def closest_point(state_function, state_domain, z):
    b = state_function(0)
    a = state_function(1) - b
    closest_x = (z[0] + z[1]*a - a*b)/(a**2 +1)
    if not (np.min(state_domain) <= closest_x <= np.max(state_domain)):
        closest_x = state_domain[np.argmin((np.abs(closest_x - state_domain[0]),
         np.abs(closest_x - state_domain[1])))]
    return closest_x, state_function(closest_x)

def get_states_in_proximity(current_state_id, state_space, epsilon):
    current_state_domain = state_space[current_state_id]['domain']
    current_state_function = state_space[current_state_id]['function']

    state_ids = list(range(len(state_space)))
    states_in_reach = list()
    for state_id in state_ids:
        state_domain = state_space[state_id]['domain']
        state_function = state_space[state_id]['function']
        distances = list()
        for x1 in current_state_domain:
            y1 = current_state_function(x1)
            point1 = np.array((x1, y1))
            for x2 in state_domain:
                    y2 = state_function(x2)
                    point2 = np.array((x2, y2))
                    distances.append(np.linalg.norm(point1 - point2))
        if np.min(distances) < epsilon:
            states_in_reach.append(state_id)
    return set(states_in_reach)

def search_for_connected_nodes(node_id, all_edges):
    connected_nodes = list()
    for edge in all_edges:
        if node_id in edge:
            i = np.where(np.array(edge) != node_id)[0][0]
            connected_nodes.append(edge[i])
    return connected_nodes

def add_reachable_edges(node_id, all_edges, reachable_edges, i, limit):
    if i < limit:
        reachable_nodes = search_for_connected_nodes(node_id, all_edges)
        connected_edges = [(node_id, reachable_node) for reachable_node in reachable_nodes]
        reversed_connected_edges = [(edge[1], edge[0]) for edge in connected_edges]
        reachable_edges.update(connected_edges)
        reachable_edges.update(reversed_connected_edges)
        for reachable_node_id in reachable_nodes:
            add_reachable_edges(reachable_node_id, all_edges, reachable_edges, i + 1, limit)
    else:
        return reachable_edges

def get_reachable_edges(starting_state_id, state_space, accepted_edges, limit): 
    node_a = state_space[starting_state_id]['edge'][0]
    node_b = state_space[starting_state_id]['edge'][1]
    reachable_edges = set(list([(node_a, node_b), (node_b, node_a)]))
    add_reachable_edges(node_a, accepted_edges, reachable_edges, 0, limit)
    add_reachable_edges(node_b, accepted_edges, reachable_edges, 0, limit)
    return reachable_edges

def emission_probabilities(z, state_space, variance):
    ep = np.empty((len(state_space)))
    for state_id in range(len(state_space)):
        state_function = state_space[state_id]['function']
        state_domain = state_space[state_id]['domain']
        closest_x, closest_y = closest_point(state_function, state_domain, z)
        ep[state_id] = 1/(np.sqrt(2*np.pi)*variance)*np.exp(-1*np.linalg.norm([z[0] - closest_x, z[1] - closest_y])/(2*variance**2))
    return ep

def transition_probabilities(state_space, edges_to_cross, required_proximity):
    tp = np.empty((len(state_space), len(state_space)))
    state_ids = list(range(len(state_space)))
    for state_id in state_ids:
        print(state_id)
        close_states = get_states_in_proximity(state_id, state_space, required_proximity)
        close_edges = [state_space[state_id]['edge'] for state_id in close_states]
        reversed_close_edges = [(edge[1], edge[0]) for edge in close_edges]
        close_edge_set = set(close_edges).union(set(reversed_close_edges))
        reachable_edges = get_reachable_edges(state_id, state_space, close_edges, edges_to_cross)
        allowed_edges = close_edge_set.intersection(reachable_edges)
        allowed_state_ids = [state['id'] for state in state_space if state['edge'] in allowed_edges]
        if len(allowed_state_ids) == 0:
            tp[state_id, state_id] = 1
        else:
            weight = 1/len(allowed_state_ids)
            tp[state_id, np.array(allowed_state_ids)] = weight
    return tp

def distance_from_endpoint_to_segment(line_domain, line_function, endpoint_domain, endpoint_function, endpoint_index):
    endpoint = np.array([endpoint_domain[endpoint_index], endpoint_function(endpoint_domain[endpoint_index])])
    x, y = closest_point(line_function, line_domain, endpoint)
    return np.linalg.norm(np.array([x, y]) - endpoint)

#Import closest_point(state_function, state_domain, z)
def distance_between_segments(state_id_a, state_id_b, state_space):
    state_a_function = state_space[state_id_a]['function']
    state_a_domain = state_space[state_id_a]['domain']
    state_b_function = state_space[state_id_b]['function']
    state_b_domain = state_space[state_id_b]['domain']
    #Assumes state_space is sorted on state_space['id'] ascending. Confirm this.
    a_to_first_endpoint_of_b = distance_from_endpoint_to_segment(state_a_domain, state_a_function,\
                                                                 state_b_domain, state_b_function, 0)
    a_to_second_endpoint_of_b = distance_from_endpoint_to_segment(state_a_domain, state_a_function,\
                                                                 state_b_domain, state_b_function, 1)
    b_to_first_endpoint_of_a = distance_from_endpoint_to_segment(state_b_domain, state_b_function,\
                                                                 state_a_domain, state_a_function, 0)
    b_to_second_endpoint_of_a = distance_from_endpoint_to_segment(state_b_domain, state_b_function,\
                                                                 state_a_domain, state_a_function, 1)
    return np.min(np.array([a_to_first_endpoint_of_b, a_to_second_endpoint_of_b, b_to_first_endpoint_of_a, b_to_second_endpoint_of_a]))   

def alternative_transition_probabilties(state_space, speed, frequency, variance, max_distance):
    n = len(state_space)
    expected_distance = speed/frequency
    tp = np.zeros((n,n))
    for i in range(n):
        dist_calc = lambda j: distance_between_segments(i, j, state_space)
        for j in range(n):
            d_ij = dist_calc(j)
            tp[i, j] = 1/(np.sqrt(2*np.pi)*variance)*np.exp(-(d_ij - expected_distance)**2/(2*variance**2))
        tp[i, :] /= np.sum(tp[i, :])
    return tp

def observation_emissions(observations, state_space, variance):
    ep = np.empty((observations.shape[0], len(state_space)))
    for i, observation in enumerate(observations):
        ep[i, :] = emission_probabilities(observation, state_space, variance)
    return ep

def forward_recursions(P, l, pi):
    n_states = P.shape[0]
    n_observations = l.shape[0]
    
    alpha = np.zeros((n_observations, n_states))
    
    alpha[0, :] = pi*l[0, :]
    C_0 = np.sum(alpha[0, :])
    alpha[0, :] /= C_0
    
    for t in range(n_observations - 1):
        for j in range(n_states):
            alpha[t+1, j] = np.sum(alpha[t, :]*P[:, j])*l[t, j]
            C_t = np.sum(alpha[t+1, :])
            alpha[t+1, :] /= C_t
    
    return alpha

def backward_recursions(P, l, alpha):
    n_states = P.shape[0]
    n_observations = l.shape[0]
    
    beta = np.zeros((n_observations, n_states))
    
    beta[n_observations - 1, :] = 1
    
    for t in range(n_observations - 2, -1, -1):
        for i in range(n_states):
            beta[t, i] = np.sum(P[i, :]*(l[t + 1, :]/np.sum(l[t+1, :]))*beta[t + 1, :])
    return beta

def viterbi(alpha, beta, P, l, pi):
    n_states = P.shape[0]
    n_observations = l.shape[0]
    
    delta = np.zeros((n_observations, n_states))
    delta[0, :] = pi*l[0, :]

    phi = np.zeros((n_observations, n_states))
    phi[0, :] = 0
    
    for t in range(1, n_observations):
        for j in range(n_states):
            delta[t, j] = np.max(delta[t-1, :]*P[:, j]*l[t, j])
            phi[t, j] = np.argmax(delta[t-1, :]*P[:, j])
    
    q_star = np.zeros((n_observations, ))
    
    P_star = np.max(delta[-1, :])
    q_star[-1] = np.argmax(delta[-1, :])
    
    for t in range(n_observations - 2, -1, -1):
        q_star[t] = phi[t+1, int(q_star[t+1])]
    
    return q_star