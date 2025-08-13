

def play(env, policy):
    state = tuple((2, 2, 4, 3, 4))   
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = policy[state] if type(state) == int else policy[tuple(state)]
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state

    return total_reward




def play_multiple_times(env, policy, max_episodes):
    total_reward = 0

    for i in range(max_episodes):
        total_reward += play(env, policy)
    
    return total_reward/max_episodes

