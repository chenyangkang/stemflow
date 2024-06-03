from functools import partial
from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.preprocessing import LabelEncoder


def make_sample_gif(
    data: pd.core.frame.DataFrame,
    file_path: str,
    col: str = "abundance",
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    figsize: Tuple[Union[float, int]] = (18, 9),
    xlims: Tuple[Union[float, int]] = None,
    ylims: Tuple[Union[float, int]] = None,
    grid: bool = True,
    lng_size: int = 20,
    lat_size: int = 20,
    xtick_interval: Union[float, int, None] = None,
    ytick_interval: Union[float, int, None] = None,
    log_scale: bool = False,
    vmin: Union[float, int] = 0.0001,
    vmax: Union[float, int, None] = None,
    lightgrey_under: bool = True,
    adder: Union[int, float] = 1,
    dpi: Union[float, int] = 300,
    fps: int = 30,
    cmap: str = "plasma",
    verbose=1,
):
    """make GIF with plt.imshow function

    A function to generate GIF file of spatio-temporal pattern.

    Args:
        data:
            Input dataframe. Data should be trimed to the target area/time slice before applying this function.
        file_path:
            Output GIF file path
        col:
            Column that contain the value to plot
        Spatio1:
            Spatio variable column 1
        Spatio2:
            Spatio variable column 2
        Temporal1:
            Temporal variable column 1
        figsize:
            Size of the figure. In matplotlib style.
        xlims:
            xlim of the figure. If None, default to xlim=(data[Spatio1].min(), data[Spatio1].max()). In matplotlib style.
        ylims:
            ylim of the figure. If None, default to ylim=(data[Spatio2].min(), data[Spatio2].max()). In matplotlib style.
        grid:
            Whether to add grids.
        lng_size:
            pixel count to aggregate at longitudinal direction. Larger means finer resolution.
        lat_size:
            pixel count to aggregate at latitudinal direction. Larger means finer resolution.
        xtick_interval:
            the size of x tick interval. If None, default to the cloest 10-based value (0.001, 0.01, 1, ... 1000, ..., etc).
        ytick_interval:
            the size of y tick interval.
        log_scale:
            log transform the target value or not.
        vmin:
            vmin of color map.
        vmax:
            vmax of color map. If None, set to the 0.9 quantile of the upper bound.
        lightgrey_under:
            Whether to set color as ligthgrey where values are below vmin.
        adder:
            If log_scale==True, value = np.log(value + adder)
        dpi:
            dpi of the GIF.
        fps:
            speed of GIF playing (frames per second).
        cmap:
            color map
        verbose:
            Print current frame if verbose >= 1.

    """
    #
    data = data.sort_values(by=Temporal1)
    data.loc[:, "Temporal_indexer"] = LabelEncoder().fit_transform(data[Temporal1])

    #
    if xlims is None:
        xlims = (data[Spatio1].min(), data[Spatio1].max())
    if ylims is None:
        ylims = (data[Spatio2].min(), data[Spatio2].max())

    #
    lng_gird = np.linspace(xlims[0], xlims[1], lng_size + 1)[1:]
    lat_gird = np.linspace(ylims[0], ylims[1], lat_size + 1)[::-1][1:]

    # xtick_interval & ytick_interval
    closest_set = [10.0 ** (i) for i in np.arange(-15, 15, 1)] + [10.0 ** (i) / 2 for i in np.arange(-15, 15, 1)]
    spatio1_base = (data[Spatio1].max() - data[Spatio1].min()) / 5
    if xtick_interval is None:
        xtick_interval = min(closest_set, key=lambda x: np.inf if x - spatio1_base > 0 else abs(x - spatio1_base))
        if xtick_interval >= 1:
            xtick_interval = int(xtick_interval)

    spatio2_base = (data[Spatio2].max() - data[Spatio2].min()) / 5
    if ytick_interval is None:
        ytick_interval = min(closest_set, key=lambda x: np.inf if x - spatio2_base > 0 else abs(x - spatio2_base))
        if ytick_interval >= 1:
            ytick_interval = int(ytick_interval)

    def round_to_same_decimal_places(A, B):
        # Convert B to string to count decimal places
        str_B = str(B)
        if "." in str_B:
            decimal_places = len(str_B.split(".")[1])
        else:
            decimal_places = 0

        # Round A to the same number of decimal places as B
        rounded_A = round(A, decimal_places)

        if abs(rounded_A) > 1000:
            # Use format to convert to scientific notation with the same number of decimal places
            formatted_A = format(rounded_A, f".{decimal_places}e")
        else:
            # Simply convert to string with the required precision
            formatted_A = f"{rounded_A:.{decimal_places}f}"

        return formatted_A

    fig, ax = plt.subplots(figsize=figsize)

    def animate(i, norm, log_scale=log_scale):
        if verbose >= 1:
            print(i, end=".")

        ax.clear()
        sub = data[data["Temporal_indexer"] == i].copy()
        temporal_value = np.array(sub[Temporal1].values)[0]

        g1 = np.digitize(sub[Spatio1], lng_gird, right=True)
        g1 = np.where(g1 >= lng_size, lng_size - 1, g1).astype("int")

        g2 = np.digitize(sub[Spatio2], lat_gird, right=True)
        g2 = np.where(g2 >= lng_size, lng_size - 1, g2).astype("int")

        sub.loc[:, f"{Spatio1}_grid"] = g1
        sub.loc[:, f"{Spatio2}_grid"] = g2
        sub = sub[(sub[f"{Spatio1}_grid"] <= lng_size - 1) & (sub[f"{Spatio2}_grid"] <= lat_size - 1)]

        sub = sub.groupby([f"{Spatio1}_grid", f"{Spatio2}_grid"])[[col]].mean().reset_index(drop=False)

        im = np.array([np.nan] * lat_size * lng_size).reshape(lat_size, lng_size)

        if log_scale:
            im[sub[f"{Spatio2}_grid"].values, sub[f"{Spatio1}_grid"].values] = np.log(sub[col] + adder)
        else:
            im[sub[f"{Spatio2}_grid"].values, sub[f"{Spatio1}_grid"].values] = sub[col]

        my_cmap = matplotlib.colormaps.get_cmap(cmap)

        if lightgrey_under:
            my_cmap.set_under("lightgrey")

        scat1 = ax.imshow(im, norm=norm, cmap=my_cmap)

        ax.set_title(f"{Temporal1}: {temporal_value}", fontsize=30)

        # Reset ticks
        xtick_labels = np.arange(xlims[0], xlims[1], xtick_interval)
        xtick_labels = [round_to_same_decimal_places(i, xtick_interval) for i in xtick_labels]
        xtick_positions = np.linspace(0, im.shape[1] - 1, len(xtick_labels))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        ytick_labels = np.arange(ylims[0], ylims[1], ytick_interval)
        ytick_labels = [round_to_same_decimal_places(i, ytick_interval) for i in ytick_labels]
        ytick_positions = np.linspace(im.shape[0] - 1, 0, len(ytick_labels))
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)

        # Grid?
        if grid:
            plt.grid(alpha=0.5)
        plt.tight_layout()

        return (scat1,)

    # scale the color norm
    if vmax is None:
        if log_scale:
            vmax = np.max(np.log(data[col].values + adder))
        else:
            vmax = np.max(data[col].values)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # for getting the color bar
    scat1 = partial(animate, norm=norm, log_scale=log_scale)(0)

    cbar = fig.colorbar(scat1[0], norm=norm, shrink=0.5)
    cbar.ax.get_yaxis().labelpad = 15
    if log_scale:
        cbar.ax.set_ylabel(f"log({col})", rotation=270)
    else:
        cbar.ax.set_ylabel(f"{col}", rotation=270)

    if grid:
        plt.grid(alpha=0.5)
    plt.tight_layout()

    partial_animate = partial(animate, norm=norm, log_scale=log_scale)

    frames = len(data["Temporal_indexer"].unique())

    # animate!
    ani = FuncAnimation(
        fig,
        partial_animate,
        interval=40,
        blit=True,
        repeat=True,
        frames=frames,
    )

    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    plt.close()
    print()
    print("Finish!")
