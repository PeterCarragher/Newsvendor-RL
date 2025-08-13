import numpy as np
import itertools
import copy

def value_iteration(env, max_iters, gamma, theta):
    v_values = np.zeros(env.observation_space.n)

    for i in range(max_iters):
        prev_v_values = np.copy(v_values)

        # Compute the value for state
        for state in range(env.observation_space.n):
            q_values = []
            # Compute the q-value for each action
            for action in range(env.action_space.n):
                q_value = 0
                # Loop through each possible outcome
                for prob, next_state, reward, _ in env.P[state][action]:
                    q_value += prob * (reward + gamma * prev_v_values[next_state])
                
                q_values.append(q_value)
            
            # Select the best action
            best_action = np.argmax(q_values)
            v_values[state] = q_values[best_action]
        
        # Check convergence
        if np.all(np.isclose(v_values, prev_v_values, atol = theta)): 
            print(f'Converged at {i}-th iteration.')
            break
    
    return v_values

def policy_extraction(env, v_values, gamma):
    policy = np.zeros(env.observation_space.n, dtype=np.int)

    # Compute the best action for each state in the game
    # Compute q-value for each (state-action) pair in the game
    for state in range(env.observation_space.n):
        q_values = []
        # Compute q_value for each action
        for action in range(env.action_space.n):
            q_value = 0
            for prob, next_state, reward, _ in env.P[state][action]:
                q_value += prob * (reward + gamma * v_values[next_state])
            q_values.append(q_value)
        
        # Select the best action
        best_action = np.argmax(q_values)
        policy[state] = best_action
    
    return policy

def value_iteration_box(env, max_iters, gamma, theta):
                
    v_values = dict()
    obs_spaces = [list(range(env.max_inventory + 1))] * env.obs_dim
    action_spaces = [list(range(env.max_order + 1))] * len(env.reorder_links) 

    for state in itertools.product(*obs_spaces):
        v_values[tuple(state)] = 0.0
    # print('vals:', v_values)

    for i in range(max_iters):
        prev_v_values = copy.deepcopy(v_values)
        # print('prev:', prev_v_values)

        for state in itertools.product(*obs_spaces):
            state = tuple(state)
            q_values = []
            for action in itertools.product(*action_spaces):
                action = tuple(action)
                # Loop through each possible outcome
                (next_state, reward) = env.MDP[state][action]
                next_reward = prev_v_values[tuple(next_state)]        
                q_values.append(reward + gamma * next_reward)
            # Select the best action
            best_action = np.argmax(q_values)
            v_values[state] = q_values[best_action]

        
        # Check convergence
        if np.all(np.isclose(np.array(list(v_values.values())), np.array(list(prev_v_values.values())), atol = theta)): 
            
            print(f'Converged at {i}-th iteration.')
            break

    return v_values

def policy_extraction_box(env, v_values, gamma):
    policy = dict()
    obs_spaces = [list(range(env.max_inventory + 1))] * env.obs_dim
    action_spaces = [list(range(env.max_order + 1))] * len(env.reorder_links) 

    for state in itertools.product(*obs_spaces):
        policy[state] = 0


    for state in itertools.product(*obs_spaces):
        q_values = dict()
        for action in itertools.product(*action_spaces):
            (next_state, reward) = env.MDP[state][action]
            q_values[action] = reward + gamma * v_values[next_state]
        
        # Select the best action
        best_action = max(q_values, key=q_values.get)
        policy[state] = best_action
    
    return policy