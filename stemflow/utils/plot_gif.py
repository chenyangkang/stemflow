from typing import Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize


def make_sample_gif(
    data: pd.DataFrame,
    file_path: str,
    col: str = "abundance",
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    figsize: Tuple[Union[float, int], Union[float, int]] = (18, 9),
    xlims: Tuple[Union[float, int], Union[float, int]] = None,
    ylims: Tuple[Union[float, int], Union[float, int]] = None,
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
    verbose: int = 1,
):
    """
    Create a GIF visualizing spatio-temporal data using plt.imshow.

    Args:
        data (pd.DataFrame): Input DataFrame, pre-filtered for the target area/time.
        file_path (str): Output GIF file path.
        col (str): Column containing the values to plot.
        Spatio1 (str): First spatial variable column.
        Spatio2 (str): Second spatial variable column.
        Temporal1 (str): Temporal variable column.
        figsize (Tuple[Union[float, int], Union[float, int]]): Figure size.
        xlims (Tuple[Union[float, int], Union[float, int]]): x-axis limits.
        ylims (Tuple[Union[float, int], Union[float, int]]): y-axis limits.
        grid (bool): Whether to display a grid.
        lng_size (int): Number of longitudinal pixels (resolution).
        lat_size (int): Number of latitudinal pixels (resolution).
        xtick_interval (Union[float, int, None]): Interval between x-ticks.
        ytick_interval (Union[float, int, None]): Interval between y-ticks.
        log_scale (bool): Whether to apply a logarithmic scale to the data.
        vmin (Union[float, int]): Minimum value for color scaling.
        vmax (Union[float, int, None]): Maximum value for color scaling.
        lightgrey_under (bool): Use light grey color for values below vmin.
        adder (Union[int, float]): Value to add before log transformation.
        dpi (Union[float, int]): Dots per inch for the output GIF.
        fps (int): Frames per second for the GIF.
        cmap (str): Colormap to use.
        verbose (int): Verbosity level.
    """
    # Sort data by the temporal variable
    data = data.sort_values(by=Temporal1)
    data["Temporal_indexer"], _ = pd.factorize(data[Temporal1])

    # Set x and y limits if not provided
    if xlims is None:
        xlims = (data[Spatio1].min(), data[Spatio1].max())
    if ylims is None:
        ylims = (data[Spatio2].min(), data[Spatio2].max())

    # Create spatial grids without slicing
    lng_grid = np.linspace(xlims[0], xlims[1], lng_size + 1)
    lat_grid = np.linspace(ylims[0], ylims[1], lat_size + 1)[::-1]

    # Determine tick intervals
    closest_set = (
        [10.0 ** i for i in np.arange(-15, 15, 1)]
        + [10.0 ** i / 2 for i in np.arange(-15, 15, 1)]
    )
    spatio1_base = (xlims[1] - xlims[0]) / 5
    if xtick_interval is None:
        xtick_interval = min(
            closest_set,
            key=lambda x: np.inf if x - spatio1_base > 0 else abs(x - spatio1_base),
        )
        if xtick_interval >= 1:
            xtick_interval = int(xtick_interval)

    spatio2_base = (ylims[1] - ylims[0]) / 5
    if ytick_interval is None:
        ytick_interval = min(
            closest_set,
            key=lambda x: np.inf if x - spatio2_base > 0 else abs(x - spatio2_base),
        )
        if ytick_interval >= 1:
            ytick_interval = int(ytick_interval)

    # Utility function to round numbers to the same decimal places
    def round_to_same_decimal_places(A, B):
        str_B = str(B)
        if "." in str_B:
            decimal_places = len(str_B.split(".")[1])
        else:
            decimal_places = 0
        rounded_A = round(A, decimal_places)
        if abs(rounded_A) > 1000:
            formatted_A = format(rounded_A, f".{decimal_places}e")
        else:
            formatted_A = f"{rounded_A:.{decimal_places}f}"
        return formatted_A

    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Set color scaling
    if vmax is None:
        vmax = (
            np.max(np.log(data[col].values + adder))
            if log_scale
            else np.max(data[col].values)
        )
        
    print(vmin, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Prepare colormap
    my_cmap = plt.get_cmap(cmap)
    if lightgrey_under:
        my_cmap.set_under("lightgrey")

    # Initialize the image to set up the colorbar
    im = ax.imshow(
        np.zeros((lat_size, lng_size)), norm=norm, cmap=my_cmap, animated=True
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.ax.get_yaxis().labelpad = 15
    cbar_label = f"log({col})" if log_scale else col
    cbar.ax.set_ylabel(cbar_label, rotation=270)

    # Precompute tick labels and positions
    x_ticks = np.arange(xlims[0], xlims[1] + xtick_interval, xtick_interval)
    x_tick_labels = [round_to_same_decimal_places(val, xtick_interval) for val in x_ticks]
    # Find positions of x_ticks within lng_grid
    x_tick_positions = np.searchsorted(lng_grid, x_ticks, side='left')
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    y_ticks = np.arange(ylims[0], ylims[1] + ytick_interval, ytick_interval)
    y_tick_labels = [round_to_same_decimal_places(val, ytick_interval) for val in y_ticks]
    # Since lat_grid is reversed, we need to account for that
    y_tick_positions = lat_size - np.searchsorted(lat_grid[::-1], y_ticks, side='left') - 1
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    # Animation function
    def animate(i):
        if verbose >= 1:
            print(f"Processing frame {i+1}/{frames}", end="\r")

        ax.clear()
        sub = data[data["Temporal_indexer"] == i]
        if sub.empty:
            return []

        temporal_value = sub[Temporal1].iloc[0]

        # Correct digitization with adjusted bins
        g1 = np.digitize(sub[Spatio1], lng_grid, right=False) - 1
        g1 = np.clip(g1, 0, lng_size - 1).astype(int)

        g2 = np.digitize(sub[Spatio2], lat_grid, right=False) - 1
        g2 = np.clip(g2, 0, lat_size - 1).astype(int)

        sub[f"{Spatio1}_grid"] = g1
        sub[f"{Spatio2}_grid"] = g2

        grouped = sub.groupby(
            [f"{Spatio2}_grid", f"{Spatio1}_grid"]
        )[col].mean()

        im_data = np.full((lat_size, lng_size), np.nan)
        indices = (grouped.index.get_level_values(0), grouped.index.get_level_values(1))
        values = np.log(grouped.values + adder) if log_scale else grouped.values
        im_data[indices] = values

        im = ax.imshow(im_data, norm=norm, cmap=my_cmap, animated=True)
        ax.set_title(f"{Temporal1}: {temporal_value}", fontsize=30)

        # Re-apply ticks and grid in each frame
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)

        if grid:
            ax.grid(alpha=0.5)

        return [im]

    frames = data["Temporal_indexer"].nunique()

    # Create animation
    ani = FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )

    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    plt.close()
    if verbose >= 1:
        print("\nAnimation saved successfully!")