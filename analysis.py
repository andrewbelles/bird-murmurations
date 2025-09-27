# !/bin/env python3  
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
    results = []
    
    for epoch, x in data.groupby('epoch'):
        points = x[['x', 'y', 'z']].to_numpy(float)
        agents = x['agent'].to_numpy()

        dist2  = np.sum((points[:, None, :] - points[None, :, :])**2, axis=2)
        np.fill_diagonal(dist2, np.inf)
        
        mins = np.sqrt(np.min(dist2, axis=1))
        results.append(pd.DataFrame({'epoch': epoch, 'agent': agents, 'minsep': mins}))

    minsep = pd.concat(results, ignore_index=True)
    minsep_over_time = minsep.groupby('epoch')['minsep'].mean()
    minsep_per_agent = minsep.groupby('agent')['minsep'].mean() 

    return minsep, minsep_over_time, minsep_per_agent 

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.file)
    alignment_over_time, alignment_per_agent, vcom_norm = alignment(data)
    minsep, minsep_over_time, minsep_per_agent = separation(data)

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


if __name__ == "__main__":
    main()
