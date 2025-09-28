# !/bin/env python3 
# 
# visualize.py  Andrew Belles  Sept 27th, 2025 
# 
# Script to visualize a full simulation from csv  
# 
# 

import argparse 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 

DT = 1e-2 

def load(path: str) -> pd.DataFrame: 

    data = pd.read_csv(path)
    needed = {"epoch", "agent", "x", "y", "z", "vx", "vy", "vz"}
    missing = needed - set(data.columns)
    if missing: 
        raise ValueError(f"CSV missing {sorted(missing)} from header")

    data["epoch"] = data["epoch"].astype(int)
    data["agent"] = data["agent"].astype(int)

    return data 


def center_of_mass(data: pd.DataFrame) -> pd.DataFrame: 

    center = data.groupby("epoch")[["x", "y", "z"]].transform("mean")
    center.columns = ["cx", "cy", "cz"]
    out = data.copy() 
    out[["cx", "cy", "cz"]] = center 
    return out 


def preprocess(data: pd.DataFrame): 

    epochs = np.sort(data["epoch"].unique())
    agents = np.sort(data["agent"].unique())
    agent_idx = {a: i for i, a in enumerate(agents)}
    T = len(epochs)
    N = len(agents)

    rel = np.zeros((T, N, 3), dtype=float)

    for i, epoch in enumerate(epochs):

        x = data[data["epoch"] == epoch].copy() 
        indices = x["agent"].map(agent_idx).to_numpy()

        P = x[["x", "y", "z"]].to_numpy(float)
        C = x[["cx", "cy", "cz"]].to_numpy(float)
        P_rel = P - C 
        rel[i, indices, :] = P_rel

    radius_max = np.max(np.linalg.norm(rel.reshape(T * N, 3), axis=1))
    if not np.isfinite(radius_max) or radius_max == 0:
        radius_max = 1.0 

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(N)]
    return epochs, rel, colors, radius_max 

def animate(epochs, rel, colors, radius_max, path=None, fps=60):

    frame_index = np.arange(0, len(epochs), 10)
    rel_view = rel[frame_index]
    epochs_view = epochs[frame_index]

    _, N, _ = rel.shape 

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Boids (Reference to C.O.M)")
    lim = radius_max * 1.05 
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatters = []
    for i in range(N):
        xi, yi, zi = rel[0, i, :]
        sc = ax.scatter([xi], [yi], [zi], s=40, depthshade=True, color=colors[i])
        scatters.append(sc)

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(j):
        P = rel_view[j]

        for i in range(N):
            xi, yi, zi = P[i]
            scatters[i]._offsets3d = ([xi], [yi], [zi])

        time = epochs_view[j] * DT 
        time_text.set_text(f"epoch={epochs_view[j]} t={time:.3f}s")
        return (*scatters, time_text)

    n_frames = rel_view.shape[0]
    anim = FuncAnimation(
        fig, update, frames=range(n_frames), 
        interval=1000/fps, blit=False
    )

    # anim.save("visualize.mp4", fps=fps, dpi=150)
    plt.show()        

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)

    args = parser.parse_args()
    data = load(args.file)
    data = center_of_mass(data)
    epochs, rel, colors, radius_max = preprocess(data)
    animate(epochs, rel, colors, radius_max)

if __name__ == "__main__":
    main()
