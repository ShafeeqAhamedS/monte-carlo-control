# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT
The Frozen Lake problem is a reinforcent Learning problem in which an agent must learn to navigatr a 16-state environwmnt in order to reach a goal state. The environment is slippery, so the agent may move in a direction other than the one intended. The agent must learn to take the shortest path to the goal state.

The environment is represented as a 4x4 grid, with each grid cell numbered as follows:

```
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15
```

### States
The environment has 16 states, numbered 0 through 15, left to right, top to bottom. The agent starts in state 0, and the goal state is state 15.
* 5 Terminal states: 5, 7, 11, 12, (15-> Goal state)
* Other are non-terminal states

### Actions
The agent has 4 possible actions:
* 0: Left
* 1: Down
* 2: Right
* 3: Up

### Transition Probabilities
Slippery surface with a 33.3% chance of moving as intended and a 66.6% chance of moving in orthogonal directions.

For example, if the agent intends to move left, there is a 
* 33.3% chance of moving left, a
* 33.3% chance of moving down, and a 
* 33.3% chance of moving up.

### Rewards
The agent receives a reward of 1 for reaching the goal state, and a reward of 0 otherwise.

### Graphical Representation



## MONTE CARLO CONTROL ALGORITHM
1. Initialize the state value function V(s) and the policy π(s) arbitrarily.
2. Generate an episode using π(s) and store the state, action and reward sequence.
3. For each state s appearing in the episode:
    * G ← return following the first occurrence of s
    * Append G to Returns(s)
    * V(s) ← average(Returns(s))
4. For each state s in the episode:
    * π(s) ← argmax_a ∑_s' P(s'|s,a)V(s')
5. Repeat steps 2-4 until the policy converges.
6. Use the function `decay_schedule` to decay the value of epsilon and alpha. The function takes the following arguments:
    * `init_value`: The initial value of epsilon or alpha.
    * `min_value`: The minimum value of epsilon or alpha.
    * `decay_ratio`: The decay ratio of epsilon or alpha.
    * `n`: The number of episodes.
    * `type`: The type of decay. It can be either `linear` or `exponential
7. Use the function `select_action` to select an action. The function takes the following arguments:
    * `state`: The current state.
    * `Q`: The Q-table.
    * `epsilon`: The value of epsilon.
    * `env`: The environment.
    * `max_steps`: The maximum number of steps.
8. Use the function `gen_traj` to generate a trajectory. The function takes the following arguments:
    * `select_action`: The function to select an action.
    * `Q`: The Q-table.
    * `epsilon`: The value of epsilon.
    * `env`: The environment.
    * `max_steps`: The maximum number of steps.
9. Use the function `tqdm` to display the progress bar. The function takes the following arguments:
    * `iterable`: The iterable to iterate over.
    * `leave`: Whether to leave the progress bar.
10. After the policy converges, use the function `np.argmax` to find the optimal policy. The function takes the following arguments:
    * `Q`: The Q-table.
    * `axis`: The axis along which to find the maximum value.


## MONTE CARLO CONTROL FUNCTION
```python
import numpy as np
from tqdm import tqdm

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n

    disc = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    def decay_schedule(init_value, min_value, decay_ratio, n):
        return np.maximum(min_value, init_value * (decay_ratio ** np.arange(n)))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    def select_action(state, Q, epsilon):
        return np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        traj = gen_traj(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool)

        for t, (state, action, reward, _, _) in enumerate(traj):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(traj[t:])
            G = np.sum(disc[:n_steps] * traj[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q

    V = np.max(Q, axis=1)
    pi = {s: np.argmax(Q[s]) for s in range(nS)}

    return Q, V, pi
```

### PROGRAM TO EVLUATE THE POLICY
```python
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123)
    np.random.seed(123)
    env.seed(123)
    results = []

    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            action = pi[state]  # Access the action for the current state from the policy dictionary
            state, _, done, _ = env.step(action)
            steps += 1
        results.append(state == goal_state)

    return np.sum(results) / len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123)
    np.random.seed(123)
    env.seed(123)
    results = []

    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            action = pi[state]  # Access the action for the current state from the policy dictionary
            state, reward, done, _ = env.step(action)
            results[-1] += reward
            steps += 1
    return np.mean(results)

def results(env,optimal_pi,optimal_V,P):
    print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, optimal_pi, goal_state=goal_state)*100,
    mean_return(env, optimal_pi)))

goal_state = 15
results(env,optimal_pi,optimal_V,P)
```
## OUTPUT:


## RESULT:
Thus a Python program is developed to find the optimal policy for the given RL environment using the Monte Carlo algorithm.