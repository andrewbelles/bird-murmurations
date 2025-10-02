# 
# summary.py  Andrew Belles  Oct 1st, 2025  
# 
# Summarizes timeseries data from simulation  
# Credit to ChatGPT 5-thinking for quick aggregation of data and plots  
# 
# 

import duckdb 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

CSV = "logging.csv"

con = duckdb.connect()
# COM, spread (radius of gyration via variances), velocity sums, speed sum
q = f"""
WITH base AS (
  SELECT epoch, x, y, z, vx, vy, vz
  FROM read_csv_auto('{CSV}', AUTO_DETECT=TRUE)
),
agg AS (
  SELECT
    epoch,
    avg(x) AS x_com, avg(y) AS y_com, avg(z) AS z_com,
    stddev_pop(x) AS sx, stddev_pop(y) AS sy, stddev_pop(z) AS sz,
    sum(vx) AS svx, sum(vy) AS svy, sum(vz) AS svz,
    sum(sqrt(vx*vx + vy*vy + vz*vz)) AS s_speed
  FROM base
  GROUP BY epoch
)
SELECT
  epoch,
  x_com, y_com, z_com,
  sqrt(sx*sx + sy*sy + sz*sz) AS Rg,
  sqrt(svx*svx + svy*svy + svz*svz) / NULLIF(s_speed,0) AS polarization
FROM agg
ORDER BY epoch
"""

df = con.execute(q).fetch_df() 

print("[PLOT] Polarization and Radius of Gyration")

f, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)
axs[0].plot(df["epoch"], df["Rg"])
axs[0].set_ylabel("Radius of gyration")
axs[1].plot(df["epoch"], df["polarization"])
axs[1].set_ylabel("Polarization")
axs[2].plot(df["epoch"], df["x_com"], label="x") 
axs[2].plot(df["epoch"], df["y_com"], label="y") 
axs[2].plot(df["epoch"], df["z_com"], label="z") 
axs[2].set_ylabel("COM")
axs[2].set_xlabel("epoch")
axs[2].legend()
f.tight_layout()
f.savefig("polar_and_rg.png", dpi=160)

STRIDE = 1

# --- FAST anisotropy in SQL (no per-epoch loop) ------------------------------
# Compute second moments & mean velocity, then R_parallel and R_perp in SQL.
q_aniso = f"""
WITH base AS (
  SELECT epoch, x, y, z, vx, vy, vz
  FROM read_csv_auto('{CSV}', AUTO_DETECT=TRUE)
),
mom AS (
  SELECT
    epoch,
    -- means
    avg(x) AS mx, avg(y) AS my, avg(z) AS mz,
    -- second moments
    avg(x*x) AS mxx, avg(y*y) AS myy, avg(z*z) AS mzz,
    avg(x*y) AS mxy, avg(x*z) AS mxz, avg(y*z) AS myz,
    -- summed velocity for direction
    sum(vx) AS svx, sum(vy) AS svy, sum(vz) AS svz
  FROM base
  GROUP BY epoch
),
covs AS (
  SELECT
    epoch,
    -- covariance entries (population)
    (mxx - mx*mx) AS cxx,
    (myy - my*my) AS cyy,
    (mzz - mz*mz) AS czz,
    (mxy - mx*my) AS cxy,
    (mxz - mx*mz) AS cxz,
    (myz - my*mz) AS cyz,
    sqrt( (svx*svx) + (svy*svy) + (svz*svz) ) AS vnorm,
    svx, svy, svz
  FROM mom
),
aniso AS (
  SELECT
    epoch,
    -- unit mean-velocity direction u (guard zero norm)
    CASE WHEN vnorm > 0 THEN svx / vnorm ELSE 1.0 END AS ux,
    CASE WHEN vnorm > 0 THEN svy / vnorm ELSE 0.0 END AS uy,
    CASE WHEN vnorm > 0 THEN svz / vnorm ELSE 0.0 END AS uz,
    cxx, cyy, czz, cxy, cxz, cyz,
    (cxx + cyy + czz) AS Rg2
  FROM covs
)
SELECT
  epoch,
  sqrt( GREATEST(0.0,
        cxx*ux*ux + cyy*uy*uy + czz*uz*uz
      + 2.0*cxy*ux*uy + 2.0*cxz*ux*uz + 2.0*cyz*uy*uz ) ) AS R_parallel,
  sqrt( GREATEST(0.0, Rg2
      - ( cxx*ux*ux + cyy*uy*uy + czz*uz*uz
        + 2.0*cxy*ux*uy + 2.0*cxz*ux*uz + 2.0*cyz*uy*uz ) ) ) AS R_perp
FROM aniso
ORDER BY epoch
"""
df_aniso = con.execute(q_aniso).fetch_df()

# --- Plot anisotropy (second figure, top panel) ------------------------------
f2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs2[0].plot(df_aniso["epoch"], df_aniso["R_parallel"], label="R_parallel")
axs2[0].plot(df_aniso["epoch"], df_aniso["R_perp"],     label="R_perp")
axs2[0].set_ylabel("spread"); axs2[0].set_title("Anisotropy of spatial spread")
axs2[0].legend()

NN_STRIDE = 1      # raise to 5 or 10 if you want it even faster
CHUNK = 1_000_000  # rows per chunk to read; tune to your RAM
need_cols = ["epoch","x","y","z"]

epochs_nn, p10s, p50s, p90s = [], [], [], []

carry = pd.DataFrame(columns=need_cols)
# Hint DuckDB can also stream, but pandas.read_csv with chunksize is simple & fast
for chunk in pd.read_csv(CSV, usecols=need_cols, chunksize=CHUNK):
    if not carry.empty:
        chunk = pd.concat([carry, chunk], ignore_index=True)
        carry = carry.iloc[0:0]

    # rows must be grouped by epoch; your logger already writes that way, but we guard via groupby
    for epoch, g in chunk.groupby("epoch", sort=True):
        # Keep last partial epoch for the next chunk (if your file ever interleaves)
        # Here we assume each epoch is complete in this group; if not, set carry = g and continue.
        # Most logger outputs are contiguous per epoch, so process directly:
        if int(epoch) % NN_STRIDE != 0:
            continue
        P = g[["x","y","z"]].to_numpy(np.float64, copy=False)

        tree = cKDTree(P)
        try:
            d, _ = tree.query(P, k=2, workers=-1)  # SciPy ≥ 1.6
        except TypeError:
            d, _ = tree.query(P, k=2)
        nn = d[:,1]

        epochs_nn.append(int(epoch))
        p10s.append(float(np.percentile(nn, 10)))
        p50s.append(float(np.percentile(nn, 50)))
        p90s.append(float(np.percentile(nn, 90)))

print("[PLOT] Nearest Neighbor Quantiles & Anisotropy")

# --- Finish second figure (bottom panel) ------------------------------------
axs2[1].plot(epochs_nn, p10s, "--", label="NN p10")
axs2[1].plot(epochs_nn, p50s,      label="NN median")
axs2[1].plot(epochs_nn, p90s, "--", label="NN p90")
axs2[1].set_xlabel("epoch"); axs2[1].set_ylabel("distance"); axs2[1].legend()

f2.tight_layout()
f2.savefig("anisotropy_nn.png", dpi=160)
