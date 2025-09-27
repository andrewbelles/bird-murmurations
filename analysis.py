#!/bin/env python3  
#
# sim_analysis.py  Andrew Belles  Sept 26th 
# 
# Analyzes a simulation data file to understand 
# boids evolution. Quantifies efficacy of simulation 
# 

import argparse 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
import seaborn as sns

MIN_DIST = 2.2 

def clean(data: pd.DataFrame):

    data = (data
            .sort_values(["epoch"])
            .drop_duplicates(["epoch", "agent"], keep="last"))
    counts = data.groupby("epoch")["agent"].nunique()
    expected_n = counts.mode().iloc[0]
    good_epochs = counts[counts == expected_n].index 
    bad_epochs  = counts[counts != expected_n]

    if len(bad_epochs):
        print(f"[WARNING] dropping {len(bad_epochs)} incomplete epochs")

    return data[data["epoch"].isin(good_epochs)].copy(), int(expected_n)


def alignment(data: pd.DataFrame):
    vcom = data.groupby('epoch')[['vx', 'vy', 'vz']].mean() 
    vcom_norm = np.linalg.norm(vcom.to_numpy(float), axis=1)
    vcom_norm = pd.Series(vcom_norm, index=vcom.index, name='vcom_norm')

    center = data.groupby('epoch')[['vx','vy','vz']].transform('mean')

    v = data[['vx', 'vy', 'vz']].to_numpy(float)
    c = center.to_numpy(float)

    vnorm = np.linalg.norm(v, axis=1)
    cnorm = np.linalg.norm(c, axis=1)
    dot   = (v * c).sum(axis=1)

    mask = (vnorm > 0) & (cnorm > 0)
    align = np.full(vnorm.shape, np.nan, dtype=float)
    align[mask] = dot[mask] / (vnorm[mask] * cnorm[mask])

    temp = data.copy() 
    temp['alignment'] = align 

    alignment_over_time = temp.groupby('epoch')['alignment'].mean()
    alignment_per_agent = temp.groupby('agent')['alignment'].mean() 

    return alignment_over_time, alignment_per_agent, vcom_norm


def separation(data: pd.DataFrame): 

    data = data.copy() 
    data["epoch"] = data["epoch"].astype(int)
    data["agent"] = data["agent"].astype(int)

    epochs = np.sort(data["epoch"].unique())
    agents = np.sort(data["agent"].unique())
    T, N = len(epochs), len(agents)

    counts = data.groupby("epoch")["agent"].nunique().reindex(
        epochs, fill_value=0).to_numpy()
    uniform = (counts == N).all()

    if not uniform:
        raise RuntimeError("Expected uniform data")

    index = pd.MultiIndex.from_product([epochs, agents], names=["epoch", "agent"])
    M = data.set_index(["epoch", "agent"]).reindex(index)
    P = M[["x", "y", "z"]].to_numpy(float).reshape(T, N, 3)

    # pairwise squared distances, distance to self is infinite 
    D2 = np.sum((P[:, :, None, :] - P[:, None, :, :])**2, axis=-1)
    index = np.arange(N)
    D2[:, index, index] = np.inf
    
    # nearest neighbor 
    nn_index = np.argmin(D2, axis=2)
    minsep = np.sqrt(np.min(D2, axis=2))

    out = pd.DataFrame({
        "epoch": np.repeat(epochs, N),
        "agent": np.tile(agents, T),
        "minsep": minsep.ravel(),
        "nn_agent": nn_index.ravel()
    })

    minsep_over_time = out.groupby("epoch")["minsep"].mean() 
    minsep_per_agent = out.groupby("agent")["minsep"].mean()
    violation_rate   = out.groupby("epoch")["minsep"].apply(
        lambda s: (s < MIN_DIST).mean())

    return {
        "minsep": out, 
        "minsep_over_time": minsep_over_time,
        "minsep_per_agent": minsep_per_agent,
        "violation_rate": violation_rate
    }

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.file)
    data, _ = clean(data)
    alignment_over_time, alignment_per_agent, vcom_norm = alignment(data)
    sep = separation(data)

    f, ax = plt.subplots(1,3,figsize=(12,5)) 
    alignment_over_time.plot(ax=ax[0], label='over time') 
    ax[0].set_title('Alignment over time')
    ax[0].set_ylabel('mean alignment (cosine)')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylim(-1, 1)
    ax[0].legend() 
    
    alignment_per_agent.sort_index().plot(ax=ax[1], kind='bar', label='per agent')
    ax[1].set_title('Alignment per agent')
    ax[1].set_ylabel('mean alignment (cosine)')
    ax[1].set_xlabel('agent')
    ax[1].set_ylim(-1, 1)
    ax[1].legend() 

    vcom_norm.plot(ax=ax[2], label='||velocity, center of mass||')
    ax[2].set_title('Center-of-mass speed')
    ax[2].set_ylabel('speed')
    ax[2].set_xlabel('epoch')
    ax[2].legend()

    f.suptitle("alignment analysis")
    f.tight_layout()
    f.savefig("alignments.png")
    plt.close() 

    f, axes = plt.subplots(2,1,figsize=(12,5))
    sep["minsep_over_time"].plot(ax=axes[0], lw=1.5)
    axes[0].axhline(MIN_DIST, ls="--", lw=1, label=f"min_dist={MIN_DIST}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("mean nearest-neighor distance")
    axes[0].set_title("Separation over time")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim(0, np.max(data["epoch"]))

    rax = axes[0].twinx()
    sep["violation_rate"].plot(ax=rax, color="tab:red", 
                               alpha=0.6, lw=1, label="violation rate")
    rax.set_ylabel("fraction below min_dist")

    data = sep["minsep"].dropna(subset=["minsep"])
    pivot = (data.pivot(index="epoch", columns="agent", values="minsep")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    axes[1].clear()
    sns.heatmap(
        pivot, 
        ax=axes[1],
        cmap="viridis",
        cbar_kws={"label": "minsep"}
    )

    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    yticks = axes[1].get_yticks() 
    yticks = yticks[(yticks >= 0) & (yticks < len(pivot.index))].astype(int)
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(pivot.index.values[yticks])
    axes[1].set_title("per-agent nearest-neighbor distance")
    axes[1].set_xlabel("agent")
    axes[1].set_ylabel("epochs")

    A = pivot.columns.to_numpy() 
    E = pivot.index.to_numpy() 
    Z = (pivot.to_numpy() < MIN_DIST).astype(float)

    axes[1].contour(A, E, Z, levels=[0.5], colors=["red"], 
                    linewidths=1, origin="upper")

    plt.suptitle("")
    f.tight_layout() 
    f.savefig("separations.png")
    plt.close()  

if __name__ == "__main__":
    main()
