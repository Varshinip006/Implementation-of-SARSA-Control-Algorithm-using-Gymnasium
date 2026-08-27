# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm in the FrozenLake-v1 environment using Gymnasium. The agent should learn an optimal policy through trial and error by updating its Q-values based on the actions selected by its current policy.


## Software Requirements

Python 3.x

Google Colab / Jupyter Notebook

Gymnasium

NumPy

Matplotlib


## Environment Description

FrozenLake-v1 is a grid-based reinforcement learning environment provided by Gymnasium. The environment consists of a 4 × 4 grid containing:

S – Starting state

F – Frozen surface where the agent can move safely

H – Hole that ends the episode with zero reward

G – Goal state that provides a reward of 1

The agent can perform four actions:

Left (0)

Down (1)

Right (2)

Up (3)

The objective is to reach the goal while avoiding the holes.

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

Create the FrozenLake-v1 environment.

Initialize the Q-table with zeros.

Set the learning rate, discount factor, and epsilon values.

Reset the environment and obtain the initial state.

Select the initial action using the epsilon-greedy policy.

Execute the selected action.

Observe the next state and reward.

Select the next action using the epsilon-greedy policy.

Update the Q-value using the SARSA update equation.

Move to the next state and action.

Repeat until the episode terminates.

Decrease epsilon gradually to reduce exploration.

Repeat the process for the specified number of episodes.

Obtain the learned state-value function and policy from the final Q-table.

Display the Q-table, value function, learned policy, and average reward.

Plot the learning curve.

## Python Program

```python

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env_desc = [
    "FFFF",
    "FHFF",
    "FFFF",
    "SFFG"
]

env = gym.make(
    "FrozenLake-v1",
    desc=env_desc,
    is_slippery=True
)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((env.observation_space.n, env.action_space.n))

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy strategy.
    """

    if np.random.rand() < epsilon:
        # Exploration
        return env.action_space.sample()
    else:
        # Exploitation
        return np.argmax(Q[state])

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    # Select initial action
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated

        # If episode is finished
        if done:

            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )

            break

        # Select next action using epsilon-greedy
        next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA update
        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        # Move to next state and action
        state = next_state
        action = next_action

    # Store reward
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

# -------------------------------------------------
# Calculate State-Value Function and Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFrozenLake Environment:")
print(np.array(env_desc))

print("\nStarting State: State 12")
print("Goal State: State 15")

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 3)
)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")

plt.grid(True)
plt.show()

# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()





```
---

## Output


Final Q-table:

<img width="297" height="386" alt="image" src="https://github.com/user-attachments/assets/6c56c103-5b73-4066-8506-c7ac40acb818" />

Estimated State-Value Function:

<img width="338" height="122" alt="image" src="https://github.com/user-attachments/assets/fd34d947-ad2f-4ba1-86d1-1a346977e50c" />

Learned Policy:

<img width="282" height="127" alt="image" src="https://github.com/user-attachments/assets/b9c2971b-291f-4444-8642-d2da2444908a" />

Average reward over last 1000 episodes: 

<img width="535" height="40" alt="image" src="https://github.com/user-attachments/assets/0f42e880-5dfb-4386-a751-1f43406611c3" />

<img width="877" height="646" alt="image" src="https://github.com/user-attachments/assets/e1500abb-63db-4d13-ba88-06e20bd5b1fd" />



---

## Result
```text

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned Q-values through repeated interaction with the environment and gradually learned a policy for reaching the goal while avoiding the holes.

```

---

## Inference
```text

The experiment demonstrates that SARSA is an on-policy reinforcement learning algorithm because the Q-value update uses the next action actually selected by the epsilon-greedy policy. Initially, the agent explores the environment using a high epsilon value. As training progresses, epsilon decreases and the agent increasingly exploits the learned Q-values. The improvement in the moving average reward indicates that the agent is learning a better policy over time.

```
---

