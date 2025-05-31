import numpy as np

class TrajectoryAgent:
    def __init__(self, arena_size=1.0, grid_size=4):
        """
        arena_size: float, size of the square arena (0 to arena_size in both x and y)
        M: int, number of grid cells per side (coarse-graining resolution)
        """
        self.arena_size = arena_size
        self.M = grid_size
        self.grid_size = arena_size / grid_size  # length of each grid cell side

    def get_grid_index(self, position):
        """
        Convert continuous position to grid cell index (row, col).
        position: tuple or array (x, y)
        returns: (row, col), zero-based indices
        """
        x, y = position
        # Clip positions inside arena bounds
        x = np.clip(x, 0, self.arena_size - 1e-9)
        y = np.clip(y, 0, self.arena_size - 1e-9)
        col = int(x // self.grid_size)
        row = int(y // self.grid_size)
        return row, col

    def get_encoded_input(self):
        """
        Return a one-hot encoded input vector of length M*M,
        representing the agent's position on an MxM grid.
        """
        row = int(self.y / self.grid_size)
        col = int(self.x / self.grid_size)
        
        # Clamp to make sure the index stays within bounds
        row = min(max(row, 0), self.M - 1)
        col = min(max(col, 0), self.M - 1)
        
        index = row * self.M + col
        encoded = np.zeros(self.M * self.M)
        encoded[index] = 1.0
        return encoded

    def get_one_hot_input(self, position):
        """
        Given continuous position, return one-hot vector (M*M length)
        with 1 at the grid cell containing the position.
        """
        row, col = self.get_grid_index(position)
        idx = row * self.M + col
        one_hot = np.zeros(self.M * self.M)
        one_hot[idx] = 1.0
        return one_hot

    def trajectory_to_inputs(self, trajectory):
        """
        trajectory: array-like shape (T, 2) with continuous positions
        returns: array shape (T, M*M) of one-hot inputs per time step
        """
        inputs = np.array([self.get_one_hot_input(pos) for pos in trajectory])
        return inputs

    def move_by_command(self, network_output, dt=1.0, step_size=0.01):
        """
        network_output: array of firing rates or spike counts from E neurons [size 4]
        Move the agent in the direction of max network output.

        Directions mapping:
        0 -> up (y+)
        1 -> down (y-)
        2 -> left (x-)
        3 -> right (x+)
        """
        direction_idx = np.argmax(network_output)
        x, y = self.position

        if direction_idx == 0:
            y += step_size * dt
        elif direction_idx == 1:
            y -= step_size * dt
        elif direction_idx == 2:
            x -= step_size * dt
        elif direction_idx == 3:
            x += step_size * dt

        # Keep agent inside arena boundaries
        x = np.clip(x, 0, self.arena_size)
        y = np.clip(y, 0, self.arena_size)

        self.position = (x, y)