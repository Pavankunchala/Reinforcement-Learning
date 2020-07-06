import gym
import numpy as np
env = gym.make('FrozenLake-v0')


def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
        # Number of evaluation iterations
        evaluation_iterations = 1
        # Initialize a value function for each state as zero
        V = np.zeros(environment.nS)
        # Repeat until change in value is below the threshold
        for i in range(int(max_iterations)):
            
                # Initialize a change of value function as zero
                delta = 0
                # Iterate though each state
                for state in range(environment.nS):
                    
                       # Initial a new value of current state
                       v = 0
                       # Try all possible actions which can be taken from this state
                       for action, action_probability in enumerate(policy[state]):
                             # Check how good next state will be
                             for state_probability, next_state, reward, terminated in environment.P[state][action]:
                                  # Calculate the expected value
                                  v += action_probability * state_probability * (reward + discount_factor * V[next_state])
                       
                       # Calculate the absolute change of value function
                       delta = max(delta, np.abs(V[state] - v))
                       # Update value function
                       V[state] = v
                evaluation_iterations += 1
                
                # Terminate if value change is insignificant
                if delta < theta:
                    
                        print(f'Policy evaluated in {evaluation_iterations} iterations.')
                        return V
 
def one_step_lookahead(environment, state, V, discount_factor):
        action_values = np.zeros(environment.nA)
        for action in range(environment.nA):
                for probability, next_state, reward, terminated in environment.P[state][action]:
                        action_values[action] += probability * (reward + discount_factor * V[next_state])
        return action_values


def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):
        # Start with a random policy
        #num states x num actions / num actions
        policy = np.ones([environment.nS, environment.nA]) / environment.nA
        # Initialize counter of evaluated policies
        evaluated_policies = 1
        # Repeat until convergence or critical number of iterations reached
        for i in range(int(max_iterations)):
                stable_policy = True
                # Evaluate current policy
                V = policy_evaluation(policy, environment, discount_factor=discount_factor)
                # Go through each state and try to improve actions that were taken (policy Improvement)
                for state in range(environment.nS):
                        # Choose the best action in a current state under current policy
                        current_action = np.argmax(policy[state])
                        # Look one step ahead and evaluate if current action is optimal
                        # We will try every possible action in a current state
                        action_value = one_step_lookahead(environment, state, V, discount_factor)
                        # Select a better action
                        best_action = np.argmax(action_value)
                        # If action didn't change
                        if current_action != best_action:
                                stable_policy = True
                                # Greedy policy update
                                policy[state] = np.eye(environment.nA)[best_action]
                evaluated_policies += 1
                # If the algorithm converged and policy is not changing anymore, then return final policy and value function
                if stable_policy:
                        print(f'Evaluated {evaluated_policies} policies.')
                        return policy, V
                    
def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
        # Initialize state-value function with zeros for each environment state
        V = np.zeros(environment.nS)
        for i in range(int(max_iterations)):
                # Early stopping condition
                delta = 0
                # Update each state
                for state in range(environment.nS):
                        # Do a one-step lookahead to calculate state-action values
                        action_value = one_step_lookahead(environment, state, V, discount_factor)
                        # Select best action to perform based on the highest state-action value
                        best_action_value = np.max(action_value)
                        # Calculate change in value
                        delta = max(delta, np.abs(V[state] - best_action_value))
                        # Update the value function for current state
                        V[state] = best_action_value
                        # Check if we can stop
                if delta < theta:
                        print(f'Value-iteration converged at iteration#{i}.')
                        break

        # Create a deterministic policy using the optimal value function
        policy = np.zeros([environment.nS, environment.nA])
        for state in range(environment.nS):
                # One step lookahead to find the best action for this state
                action_value = one_step_lookahead(environment, state, V, discount_factor)
                # Select best action based on the highest state-action value
                best_action = np.argmax(action_value)
                # Update the policy to perform a better action at a current state
                policy[state, best_action] = 1.0
        return policy, V
    
    
    
def play_episodes(environment, n_episodes, policy):
        wins = 0
        total_reward = 0
        for episode in range(n_episodes):
                terminated = False
                state = environment.reset()
                while not terminated:
                        # Select best action to perform in a current state
                        action = np.argmax(policy[state])
                        # Perform an action an observe how environment acted in response
                        next_state, reward, terminated, info = environment.step(action)
                        # Summarize total reward
                        total_reward += reward
                        # Update current state
                        state = next_state
                        # Calculate number of wins over episodes
                        if terminated and reward == 1.0:
                                wins += 1
        average_reward = total_reward / n_episodes
        return wins, total_reward, average_reward

# Number of episodes to play
n_episodes = 10000
# Functions to find best policy
solvers = [('Policy Iteration', policy_iteration),
           ('Value Iteration', value_iteration)]
for iteration_name, iteration_func in solvers:
        # Load a Frozen Lake environment
        environment = gym.make('FrozenLake-v0')
        # Search for an optimal policy using policy iteration
        policy, V = iteration_func(environment.env)
        # Apply best policy to the real environment
        wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
        print(f'{iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
        print(f'{iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    