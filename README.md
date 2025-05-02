# Optimal Policies with Dynamic Programming – Gridworld Parking MDP

This project implements **Dynamic Programming** algorithms to solve a parking optimization problem modeled as a **Markov Decision Process (MDP)**. The environment simulates **Gridworld City**, where the city council aims to optimize parking prices to maximize social welfare using reinforcement learning principles.
---
## Project Context

In Gridworld City:
- **States** represent the number of occupied parking spots.
- **Actions** represent the price charged for parking.
- **Rewards** are higher when more spots are used, except when all are occupied (to discourage full saturation).

The objective is to find an **optimal pricing policy** using:
- **Policy Evaluation**
- **Policy Iteration**
- **Value Iteration**

### Algorithms Implemented

- **Policy Evaluation** – Evaluates a given policy using the Bellman Expectation Equation.
- **Policy Iteration** – Alternates between policy evaluation and improvement until convergence.
- **Value Iteration** – Directly computes the optimal value function using the Bellman Optimality Equation.
- **Greedy Policy Extraction** – Constructs a policy by acting greedily with respect to current value estimates.
---

## Project Structure

├── gridworld_dp.py         # Main file: all implementations for evaluation, iteration, and visualization
├── tools.py                # Provided environment simulating the Gridworld parking scenario
├── grader.py               # Provided helper for value checking and autograding
├── README.md               # Project documentation
