import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec


def build_segments(x, z):
    """
    Convert trajectory points into line segments for LineCollection.
    x, z: shape [T]
    return: segments of shape [T-1, 2, 2]
    """
    points = np.array([x, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def compute_error(df):
    """
    Compute instantaneous x-z Euclidean tracking error.
    """
    x = df["x"].to_numpy()
    z = df["z"].to_numpy()
    x_ref = df["x_ref"].to_numpy()
    z_ref = df["z_ref"].to_numpy()

    min_len = min(len(x), len(z), len(x_ref), len(z_ref))
    x = x[:min_len]
    z = z[:min_len]
    x_ref = x_ref[:min_len]
    z_ref = z_ref[:min_len]

    err = np.sqrt((x - x_ref) ** 2 + (z - z_ref) ** 2)
    return x, z, x_ref, z_ref, err


def plot_colored_trajectory(ax, x, z, x_ref, z_ref, err, norm, cmap="turbo", lw=2.5):
    """
    Plot actual trajectory colored by error, plus black dashed reference.
    """
    segments = build_segments(x, z)

    err_seg = 0.5 * (err[:-1] + err[1:])
    err_seg = np.maximum(err_seg, 1e-6)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(err_seg)
    lc.set_linewidth(lw)
    ax.add_collection(lc)

    ax.plot(x_ref, z_ref, "k--", linewidth=1.5, alpha=0.9)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    return lc


def load_one_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return compute_error(df)


if __name__ == "__main__":
    files = {
        ("Figure-8 | Linear", "Nominal MPC"): "results_tracking_figure8/Nominal_MPC_linear_slope0.0001_noise0.0005_seed51.csv",
        ("Figure-8 | Linear", "Neural MPC"): "results_tracking_figure8/Neural_MPC_linear_slope0.0001_noise0.0005_seed51.csv",
        ("Figure-8 | Linear", "T2S-MPC (ours)"): "results_tracking_figure8/T2S_MPC_linear_slope0.0001_noise0.0005_seed51.csv",

        ("Figure-8 | Periodic", "Nominal MPC"): "results_tracking_figure8/Nominal_MPC_periodic_A0.003_period2.0_noise0.0005_seed42.csv",
        ("Figure-8 | Periodic", "Neural MPC"): "results_tracking_figure8/Neural_MPC_periodic_A0.003_period2.0_noise0.0005_seed42.csv",
        ("Figure-8 | Periodic", "T2S-MPC (ours)"): "results_tracking_figure8/T2S_MPC_periodic_A0.003_period2.0_noise0.0005_seed42.csv",

        ("Circle | Linear", "Nominal MPC"): "results_tracking_circle/Nominal_MPC_linear_slope0.0001_noise0.0005_seed42.csv",
        ("Circle | Linear", "Neural MPC"): "results_tracking_circle/Neural_MPC_linear_slope0.0001_noise0.0005_seed42.csv",
        ("Circle | Linear", "T2S-MPC (ours)"): "results_tracking_circle/T2S_MPC_linear_slope0.0001_noise0.0005_seed42.csv",

        ("Circle | Periodic", "Nominal MPC"): "results_tracking_circle/Nominal_MPC_periodic_A0.003_period2.0_noise0.0005_seed51.csv",
        ("Circle | Periodic", "Neural MPC"): "results_tracking_circle/Neural_MPC_periodic_A0.003_period2.0_noise0.0005_seed51.csv",
        ("Circle | Periodic", "T2S-MPC (ours)"): "results_tracking_circle/T2S_MPC_periodic_A0.003_period2.0_noise0.0005_seed51.csv",
    }

    # row_groups = [
    #     ("Circle | Linear", "Circle | Periodic"),
    #     ("Figure-8 | Linear", "Figure-8 | Periodic"),
    # ]

    row_groups = [
        ("Circle | Linear", "Circle | Periodic"),
        ("Figure-8 | Linear", "Figure-8 | Periodic"),
    ]

    # Define simple names for the y-axis labels for each row
    row_display_names = ["Circle", "Figure-8"]

    col_names = ["Nominal MPC", "Neural MPC", "T2S-MPC (ours)"]

    data = {}
    all_err_seg = []

    for key, path in files.items():
        x, z, x_ref, z_ref, err = load_one_csv(path)
        data[key] = {
            "x": x,
            "z": z,
            "x_ref": x_ref,
            "z_ref": z_ref,
            "err": err,
        }

        err_seg = 0.5 * (err[:-1] + err[1:])
        err_seg = np.maximum(err_seg, 1e-6)
        all_err_seg.append(err_seg)

    all_err_seg = np.concatenate(all_err_seg)

    norm = LogNorm(vmin=1e-3, vmax=1e-1)

    X_LIM = (-0.55, 0.55)
    Z_LIM = (0.45, 1.55)

    # =============================
    # Font control
    # =============================
    FONT_SIZE = 17

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE
    })

    fig = plt.figure(figsize=(22, 8))

    gs = GridSpec(
        2, 7,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.05],
        wspace=0.15,
        hspace=0.08
    )

    axes = np.empty((2, 6), dtype=object)
    for i in range(2):
        for j in range(6):
            axes[i, j] = fig.add_subplot(gs[i, j])

    for i, (left_group, right_group) in enumerate(row_groups):
        # Left block: columns 0,1,2
        for j, col_name in enumerate(col_names):
            ax = axes[i, j]
            d = data[(left_group, col_name)]

            plot_colored_trajectory(
                ax=ax,
                x=d["x"],
                z=d["z"],
                x_ref=d["x_ref"],
                z_ref=d["z_ref"],
                err=d["err"],
                norm=norm,
                cmap="turbo",
                lw=5.0,
            )

            ax.set_xlim(X_LIM)
            ax.set_ylim(Z_LIM)

            if i == 0:
                ax.set_title(col_name, fontsize=13, pad=8)

            # Only first subplot of left block shows y label/ticks
            if j == 0:
                ax.set_ylabel(f"{row_display_names[i]}\nz [m]")
            else:
                ax.set_ylabel("")
                ax.tick_params(left=False, labelleft=False)

            # Only bottom row shows x label/ticks
            if i == len(row_groups) - 1:
                ax.set_xlabel("x [m]")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

        # Right block: columns 3,4,5
        for j, col_name in enumerate(col_names):
            ax = axes[i, j + 3]
            d = data[(right_group, col_name)]

            plot_colored_trajectory(
                ax=ax,
                x=d["x"],
                z=d["z"],
                x_ref=d["x_ref"],
                z_ref=d["z_ref"],
                err=d["err"],
                norm=norm,
                cmap="turbo",
                lw=5.0,
            )

            ax.set_xlim(X_LIM)
            ax.set_ylim(Z_LIM)

            if i == 0:
                ax.set_title(col_name, fontsize=13, pad=8)

            # Remove y labels/ticks from the whole right block,
            # but keep the vertical spine visible
            ax.set_ylabel("")
            ax.tick_params(left=False, labelleft=False)

            # Only bottom row shows x label/ticks
            if i == len(row_groups) - 1:
                ax.set_xlabel("x [m]")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    fig.text(0.245, 0.97, "Linear Disturbance", ha="center", va="top", fontsize=15)
    fig.text(0.69, 0.97, "Periodic Disturbance", ha="center", va="top", fontsize=15)

    from matplotlib.ticker import FixedLocator, FixedFormatter

    sm = ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([])
    cax = fig.add_subplot(gs[:, 6])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Tracking error [m]")

    tick_vals = [1e-3, 1e-2, 1e-1]
    tick_labels = [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]

    cbar.ax.yaxis.set_major_locator(FixedLocator(tick_vals))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))

    os.makedirs("results_plot_final", exist_ok=True)
    out_path = "results_plot_final/figure8_circle_error_colormap_logscale_final.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    plt.show()