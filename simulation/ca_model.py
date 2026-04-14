"""Cellular Automata fire spread simulation."""

import numpy as np
import imageio
import os


def simulate_fire(prediction, steps=10, threshold=None):
    """
    Simulate fire spread using cellular automata.

    States: 0=unburned, 1=burning, 2=burned out
    Uses the saved optimal threshold if available, else 0.5.
    """
    if threshold is None:
        t_path = "outputs/evaluation/best_threshold.txt"
        if os.path.exists(t_path):
            threshold = float(open(t_path).read().strip())
        else:
            threshold = 0.5

    grid = np.zeros_like(prediction, dtype=int)
    burn_time = np.zeros_like(prediction)
    grid[prediction > threshold] = 1  # initial fire

    frames = [grid.copy()]
    H, W = grid.shape

    for _ in range(steps):
        new_grid = grid.copy()
        new_burn = burn_time.copy()

        for i in range(H):
            for j in range(W):
                if grid[i, j] == 1:
                    new_burn[i, j] += 1
                    # Spread to 8-neighbors
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0:
                                if np.random.rand() < prediction[ni, nj] * 2.0:
                                    new_grid[ni, nj] = 1
                    # Burn out after 3 steps
                    if new_burn[i, j] > 3:
                        new_grid[i, j] = 2

        grid, burn_time = new_grid, new_burn
        frames.append(grid.copy())

    return frames


def simulate_steps(prediction, steps_list=(1, 2, 3, 6, 12)):
    """Run simulation and return specific time-step frames."""
    frames = simulate_fire(prediction, steps=max(steps_list))
    return {s: frames[s] for s in steps_list}


def save_animation(prediction, steps=12, path="outputs/animations/fire.gif"):
    """Save simulation as GIF."""
    frames = simulate_fire(prediction, steps)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = [(f / 2.0 * 255).astype(np.uint8) for f in frames]
    imageio.mimsave(path, images, duration=0.5)
    return path


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pred = np.load("outputs/predictions/fire_map.npy")
    frames = simulate_fire(pred, steps=10)

    for i, frame in enumerate(frames):
        plt.clf()
        plt.imshow(frame, cmap="hot", vmin=0, vmax=2)
        plt.title(f"Step {i}")
        plt.pause(0.5)
    plt.show()