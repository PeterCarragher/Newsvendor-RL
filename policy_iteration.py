import numpy as np
import itertools
import copy

def policy_evaluation(env, policy, max_iters, gamma, theta):
    v_values = np.zeros(env.observation_space.n)

    for i in range(max_iters):
        prev_v_values = np.copy(v_values)

        # Compute the value for state
        for state in range(env.observation_space.n):
            # Compute the q-value for each action
            action = policy[state]
            q_value = 0
                # Loop through each possible outcome
            for prob, next_state, reward, done in env.P[state][action]:
                q_value += prob * (reward + gamma * prev_v_values[next_state])
                
            
            # Select the best action
            v_values[state] = q_value
        
        # Check convergence
        if np.all(np.isclose(v_values, prev_v_values, atol=theta)):
#             print(f'Converged at {i}-th iteration.')
            break
    
    return v_values


def policy_improvement(env, old_policy, old_v_values, gamma):
    policy = old_policy.copy() #np.zeros(env.observation_space.n, dtype=np.int)
        # Compute the value for state
    for state in range(env.observation_space.n):
        q_values = []
            # Compute the q-value for each action
        for action in range(env.action_space.n):
            q_value = 0
                # Loop through each possible outcome
            for prob, next_state, reward, done in env.P[state][action]:
                q_value += prob * (reward + gamma * old_v_values[next_state])
                
            q_values.append(q_value)
            
            # Select the best action
        best_action = np.argmax(q_values)
        policy[state] = best_action
        
        # Check convergence
    return policy

def policy_iteration(env, max_iters, gamma, theta):
    policy = np.random.randint(env.action_space.n, size=env.observation_space.n, dtype=np.int)
    
    for i in range(max_iters):
        v_values = policy_evaluation(env, policy, max_iters=1000, gamma=gamma, theta=theta)
        new_policy = policy_improvement(env, policy, v_values, gamma=gamma)
        if (np.array_equal(policy, new_policy)):
            print(f'Converged at {i}-th iteration.')
            break
        policy = new_policy.copy()
    return policy

# Special case for Supply Chain Network action & obs spaces
def policy_evaluation_box(env, policy, max_iters, gamma, theta):
    v_values = dict()
    obs_spaces = [list(range(env.max_inventory + 1))] * env.obs_dim

    for state in itertools.product(*obs_spaces):
        v_values[tuple(state)] = 0.0

    for i in range(max_iters):
        prev_v_values = copy.deepcopy(v_values)

        for state in itertools.product(*obs_spaces):
            state = tuple(state)
            action = tuple(policy[state])
            # print(action)
            (next_state, reward) = env.MDP[state][action]
            next_reward = prev_v_values[tuple(next_state)]        
            v_values[state] = reward + gamma * next_reward

        if np.all(np.isclose(np.array(list(v_values.values())), np.array(list(prev_v_values.values())), atol = theta)): 
            # print(f'Converged at {i}-th iteration.')
            break
    
    return v_values


def policy_improvement_box(env, old_policy, old_v_values, gamma):
    policy = copy.deepcopy(old_policy) 

    obs_spaces = [list(range(env.max_inventory + 1))] * env.obs_dim
    action_spaces = [list(range(env.max_order + 1))] * len(env.reorder_links) 

    for state in itertools.product(*obs_spaces):
        q_values = dict()
        # Compute the q-value for each action
        for action in itertools.product(*action_spaces):
            next_state, reward = env.MDP[tuple(state)][tuple(action)]                
            q_values[action] = reward + gamma * old_v_values[next_state]
            
        # Select the best action
        best_action = max(q_values, key=q_values.get)
        policy[state] = best_action
        
    return policy



def policy_iteration_box(env, max_iters, gamma, theta):
    policy = dict()
    obs_spaces = [list(range(env.max_inventory + 1))] * env.obs_dim
    # action_spaces = [list(range(env.max_order + 1))] * len(env.reorder_links) 
    null_action = tuple([0] * len(env.reorder_links))
    for state in itertools.product(*obs_spaces):
        policy[state] = null_action

    for i in range(max_iters):
        v_values = policy_evaluation_box(env, policy, max_iters=1000, gamma=gamma, theta=theta)
        new_policy = policy_improvement_box(env, policy, v_values, gamma=gamma)
        if (np.array_equal(policy, new_policy)):
            print(f'Converged at {i}-th iteration.')
            break
        policy = copy.deepcopy(new_policy)
    return policy