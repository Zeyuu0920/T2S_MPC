import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FormatStrFormatter

base_dir = "results_stabilization"

variants = {
 
    "Nominal MPC": "Nominal_MPC_periodic_A0.003_period2.0_noise0.0005_seed",
    "Neural MPC": "Neural_MPC_periodic_A0.003_period2.0_noise0.0005_seed",
    "T2S-MPC(w/o time emd)": "T2S_wo_time_emd_periodic_A0.003_period2.0_noise0.0005_seed",
    "T2S-MPC(w/o 2 scales)": "T2S_wo_two_scales_periodic_A0.003_period2.0_noise0.0005_seed",
    "T2S-MPC(ours)": "T2S_MPC_periodic_A0.003_period2.0_noise0.0005_seed",

}

plot_path = "output_final/stabilization_error_periodic.png"


FONT_SIZE = 25
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE
})


fig, ax = plt.subplots(figsize=(15, 10))

axins = inset_axes(
    ax,
    width="50%",
    height="50%",
    loc="upper right",
    borderpad=1.2
)
axins.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


inset_curves = []


for label, prefix in variants.items():
    pattern = os.path.join(base_dir, f"{prefix}*.csv")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"Warning: No files found for {label} with pattern {pattern}. Skipping this variant.")
        continue

    all_errors = []
    time_arrays = []

    for file in files:
        df = pd.read_csv(file)

        required_cols = ["time", "x", "x_ref", "z", "z_ref"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: {file} missing columns {missing_cols}. Skipping this file.")
            continue

        # 计算 X-Z error
        x_err = df["x"].to_numpy() - df["x_ref"].to_numpy()
        z_err = df["z"].to_numpy() - df["z_ref"].to_numpy()
        xz_err = np.sqrt(x_err**2 + z_err**2)

        all_errors.append(xz_err)
        time_arrays.append(df["time"].to_numpy())

    if len(all_errors) == 0:
        print(f"Warning: No valid files for {label}. Skipping this variant.")
        continue


    min_len = min(len(arr) for arr in all_errors)
    all_errors = np.array([arr[:min_len] for arr in all_errors])
    time_arrays = np.array([arr[:min_len] for arr in time_arrays])


    t = time_arrays[0]


    if not np.allclose(time_arrays, t[None, :], atol=1e-8):
        print(f"Warning: Time arrays are not identical for {label}. Using the first run's time array.")

    per_run_mean = np.mean(all_errors, axis=1)
    avg_error = np.mean(per_run_mean)
    std_error = np.std(per_run_mean)

    print(f"{label}: Mean X-Z Error = {avg_error:.6f} \u00B1 {std_error:.6f}")

    mean = np.mean(all_errors, axis=0)
    std = np.std(all_errors, axis=0)


    ax.plot(t, mean, label=label, linewidth=2.5)
    ax.fill_between(t, mean - std, mean + std, alpha=0.2)


    axins.plot(t, mean, linewidth=2.5)
    axins.fill_between(t, mean - std, mean + std, alpha=0.2)

    inset_curves.append((t, mean, std))


ax.set_xlabel("Time [s]")
ax.set_ylabel("X-Z Error [m]")
ax.tick_params(axis='both', labelsize=FONT_SIZE)
ax.legend(loc='upper left', fontsize=FONT_SIZE - 1, frameon=True)
ax.grid(True, alpha=0.3)

x1, x2 = 8, 20

if len(inset_curves) > 0:
    ymins = []
    ymaxs = []

    for t, mean, std in inset_curves:
        mask = (t >= x1) & (t <= x2)
        if np.any(mask):
            ymins.append(np.min(mean[mask] - std[mask]))
            ymaxs.append(np.max(mean[mask] + std[mask]))

    if len(ymins) > 0 and len(ymaxs) > 0:
        y1 = max(0.0, min(ymins))  
        y2 = max(ymaxs)


        margin = 0.05 * (y2 - y1) if y2 > y1 else 0.001
        # y1 = max(0.0, y1 - margin)
        y1 = -0.03
        y2 = y2 + margin

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
    else:
        print("Warning: No data found in the inset x-range.")
else:
    print("Warning: No valid curves were plotted.")

axins.tick_params(axis='both', labelsize=FONT_SIZE - 1)
axins.grid(True, alpha=0.3)
axins.patch.set_alpha(0.9)


mark_inset(
    ax,
    axins,
    loc1=3,   # lower left
    loc2=4,   # lower right
    fc="none",
    ec="0.5",
    lw=1.0
)

plt.tight_layout()

os.makedirs("output_final", exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {plot_path}")

plt.show()