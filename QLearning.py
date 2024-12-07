import random
import numpy as np

grid= np.array([
[0, -1, 0, 10],
    [0, 0, 0, -1],
    [0, -1, 0, 0],
    [0, 0, 0, 0]
])

gamma=0.9
living_rewards= -0.01
transition_probabilitis= {"success":0.8, "right":0.1, "left": 0.1}

def step(state,action):
    r,c= state
    if action==0: #bala
        return max(0,r-1),c
    elif action==1: #rast
        return r,min(c+1, grid.shape[1] - 1)
    elif action==2:#payiin
        return min(r+1, grid.shape[0]-1),c
    elif action ==3: #chap
        return r,max(0,c-1)


def value_iteration(grid, gamma,transition_porbabilitis, iterations= 100):
    number_rows,number_columns= grid.shape
    V=np.zeros_like(grid,dtype=float)
    policy= np.zeros_like(grid,dtype=int)
    for _ in range(iterations):
        new_V= np.copy(V)
        for r in range(number_rows):
            for c in range(number_columns):
                if grid[r,c] != 0: #khoone haye payan taqiri nmikone
                    new_V[r,c]= grid[r,c]
                    continue
                q_values=[]
                for action in range(4): #4 ta action darim dg
                    total= 0
                    for probabilitis, offset in zip(transition_probabilitis.values(),[-1,0,1]):
                        next_action = (action + offset) % 4
                        next_state = step((r, c), next_action)
                        total += probabilitis * (grid[next_state] + gamma * V[next_state])
                    q_values.append(total)
                    new_V[r, c] = max(q_values)
                    policy[r, c] = np.argmax(q_values)
            if np.max(np.abs(new_V - V)) < 1e-3:
                break
        V = new_V
    return V, policy

def policy_iterations(grid, gamma,transition_probabilitis, iterations=100):
    number_rows,number_columns= grid.shape
    V=np.zeros_like(grid,dtype=float)
    policy= np.zeros_like(grid,dtype=int)
    for _ in range(iterations):
        for _ in range(50):
            new_V=np.copy(V)
            for r in range(number_rows):
                for c in range(number_columns):
                    if grid[r,c] !=0:
                        new_V[r,c]= grid[r,c]
                        continue
                    action = policy[r,c]
                    total=0
                    for probabilitis, offset in zip(transition_probabilitis.values(), [-1,0,1]):
                        next_action = (action + offset) % 4
                        next_state = step((r, c), next_action)
                        total += probabilitis * (grid[next_state] + gamma * V[next_state])
                    new_V[r, c] = total
                    if np.max(np.abs(new_V - V)) < 1e-3:
                        break
                    V = new_V

                policy_stable = True
                for r in range(number_rows):
                    for c in range(number_columns):
                        if grid[r, c] != 0:  #khoonehaye payani taqir nmikone
                            continue
                        q_values = []
                        for action in range(4):
                            total = 0
                            for probabilitis, offset in zip(transition_probabilitis.values(), [-1, 0, 1]):
                                next_action = (action + offset) % 4
                                next_state = step((r, c), next_action)
                                total += probabilitis * (grid[next_state] + gamma * V[next_state])
                            q_values.append(total)
                        best_action = np.argmax(q_values)
                        if policy[r, c] != best_action:
                            policy_stable = False
                        policy[r, c] = best_action
                if policy_stable:
                    break
            return V, policy



def simulate(policy, grid, episodes=100):
    total_reward = 0
    for _ in range(episodes):
        state = (np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1]))
        episode_reward = 0
        for _ in range(100):
            r, c = state
            action = policy[r, c]
            next_state = step(state, action)
            episode_reward += grid[next_state] + living_rewards
            state = next_state
            if grid[state] != 0:  # be khoneye payan resiidan
                break
        total_reward += episode_reward
    return total_reward / episodes

def print_policy(policy):
    action_symbols = ['↑', '→', '↓', '←']
    for row in policy:
        print(" ".join(action_symbols[a] for a in row))

V_vi, policy_vi = value_iteration(grid, gamma, transition_probabilitis)
mean_reward_vi = simulate(policy_vi, grid)
print("value function (value iterations):")
print(V_vi)
print("\n optimal policy (value iterations):")
print_policy(policy_vi)


V_pi, policy_pi = policy_iterations(grid, gamma, transition_probabilitis)
mean_reward_pi = simulate(policy_pi, grid)
print("\n value function (policy iteration):")
print(V_pi)
print("\n optimal policy (policy iteration):")
print_policy(policy_pi)

# *********************** natijeye nahayi
print("\n mean rewards (value iteration):", mean_reward_vi)
print("mean reward (policy iteration):", mean_reward_pi)