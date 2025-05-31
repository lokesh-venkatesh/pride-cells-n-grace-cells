import numpy as np
import matplotlib.pyplot as plt

from trajectory_agent import TrajectoryAgent
from ei_network import EINetwork

# --- Simulation parameters ---
arena_size = 1.0
grid_size = 4
num_timesteps = 500
dt = 1.0
step_size = 0.05  # Movement step size

# --- Initialize agent and network ---
agent = TrajectoryAgent(arena_size=arena_size, grid_size=grid_size)
agent.position = (0.5, 0.5)  # Starting position at the center

network = EINetwork(num_exc=4, num_inh_groups=1, num_inh_per_group=1)

# --- Logging ---
trajectory = [agent.position]
rewards = []

# --- Main loop ---
for t in range(num_timesteps):
    # Encode agent position as one-hot input
    encoded_input = agent.get_one_hot_input(agent.position)

    # Optionally scale input (e.g., to match network input scale)
    external_input = encoded_input[:4]  # Use first 4 inputs to match num_exc (basic simplification)

    # Simulate network step
    reward = 0.0  # Placeholder: reward can be defined based on task or environment
    network.step(ext_input=external_input, reward=reward, dt=dt)

    # Get E neuron outputs (e.g., firing rates or spikes)
    firing_rates = network.get_E_firing_rates(window=10)

    # Move agent based on network output
    agent.move_by_command(firing_rates, dt=dt, step_size=step_size)

    # Log position and reward
    trajectory.append(agent.position)
    rewards.append(reward)

# --- Plotting trajectory ---
trajectory = np.array(trajectory)

plt.figure(figsize=(6, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', markersize=2)
plt.title("Agent Trajectory in Arena")
plt.xlim(0, arena_size)
plt.ylim(0, arena_size)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# --- Plot network activity ---
network.plot_activity()
