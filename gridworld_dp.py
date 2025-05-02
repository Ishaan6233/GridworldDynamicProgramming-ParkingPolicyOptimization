import numpy as np
import tools

# ------------------
# Policy Evaluation
# ------------------
def bellman_update(env, V, pi, s, gamma):
    V[s] = sum(
        pi[s][a] * sum(
            transitions[i, 1] * (transitions[i, 0] + gamma * V[i])
            for i in range(len(transitions))
        )
        for a in range(len(pi[s]))
        for transitions in [env.transitions(s, a)]
    )

def evaluate_policy(env, V, pi, gamma, theta):
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in env.S:
            last_v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(last_v - V[s]))
    return V

# ------------------
# Policy Iteration
# ------------------
def q_greedify_policy(env, V, pi, s, gamma):
    q_values = np.zeros(len(env.A))
    for a in env.A:
        q_values[a] = sum(
            p * (r + gamma * V[int(sp)])
            for sp, (r, p) in enumerate(env.transitions(s, a))
        )
    max_q = np.max(q_values)
    greedy_actions = [a for a, q in enumerate(q_values) if q == max_q]
    for a in env.A:
        pi[s][a] = 1 / len(greedy_actions) if a in greedy_actions else 0

def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False
    return pi, policy_stable

def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return V, pi

# ------------------
# Value Iteration
# ------------------
def bellman_optimality_update(env, V, s, gamma):
    V[s] = max(
        sum(
            p * (r + gamma * V[int(sp)])
            for sp, (r, p) in enumerate(env.transitions(s, a))
        )
        for a in env.A
    )

def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi

# ------------------
# Alternative Value Iteration
# ------------------
def value_iteration2(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            q_greedify_policy(env, V, pi, s, gamma)
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, pi

# ------------------
# Run Example
# ------------------
if __name__ == "__main__":
    env = tools.ParkingWorld(num_spaces=10, num_prices=4)
    gamma = 0.9
    theta = 0.1

    # Policy Iteration
    V_pi, pi_pi = policy_iteration(env, gamma, theta)
    print("Policy Iteration - Value Function:")
    print(V_pi)
    tools.plot(V_pi, pi_pi)

    # Value Iteration
    V_vi, pi_vi = value_iteration(env, gamma, theta)
    print("Value Iteration - Value Function:")
    print(V_vi)
    tools.plot(V_vi, pi_vi)

    # Value Iteration with Policy
    V_vi2, pi_vi2 = value_iteration2(env, gamma, theta)
    print("Value Iteration 2 - Value Function:")
    print(V_vi2)
    tools.plot(V_vi2, pi_vi2)
