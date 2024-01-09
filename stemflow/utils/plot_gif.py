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
    xlims: Tuple[Union[float, int]] = (-180, 180),
    ylims: Tuple[Union[float, int]] = (-90, 90),
    grid: bool = True,
    lng_size: int = 360,
    lat_size: int = 180,
    xtick_interval: Union[float, int] = 30,
    ytick_interval: Union[float, int] = 30,
    log_scale: bool = False,
    vmin: Union[float, int] = 0.0001,
    vmax: Union[float, int] = None,
    lightgrey_under: bool = True,
    adder: Union[int, float] = 1,
    dpi: Union[float, int] = 300,
    fps: int = 30,
    cmap: str = "plasma",
):
    """make GIF with plt.imshow function

    A function to generate GIF file of spatio-temporal pattern.

    Args:
        data:
            Input dataframe
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
            xlim of the figure. In matplotlib style.
        ylims:
            ylim of the figure. In matplotlib style.
        grid:
            Whether to add grids.
        lng_size:
            pixel count to aggregate at longitudinal direction. Larger means finer resolution.
        lat_size:
            pixel count to aggregate at latitudinal direction. Larger means finer resolution.
        xtick_interval:
            the size of x tick interval.
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

    """
    #
    data = data.sort_values(by=Temporal1)
    data["Temporal_indexer"] = LabelEncoder().fit_transform(data[Temporal1])

    #
    lng_gird = np.linspace(xlims[0], xlims[1], lng_size)
    lat_gird = np.linspace(ylims[0], ylims[1], lat_size)[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    def animate(i, norm, log_scale=log_scale):
        print(i, end=".")
        ax.clear()
        sub = data[data["Temporal_indexer"] == i]
        temporal_value = np.array(sub[Temporal1].values)[0]

        sub[f"{Spatio1}_grid"] = np.digitize(sub[Spatio1], lng_gird, right=True)
        sub[f"{Spatio2}_grid"] = np.digitize(sub[Spatio2], lat_gird, right=False)
        sub = sub[(sub[f"{Spatio1}_grid"] <= lng_size - 1) & (sub[f"{Spatio2}_grid"] <= lat_size - 1)]

        sub = sub.groupby([f"{Spatio1}_grid", f"{Spatio2}_grid"])[[col]].mean().reset_index(drop=False)

        im = np.array([np.nan] * lat_size * lng_size).reshape(lat_size, lng_size)

        if log_scale:
            im[sub[f"{Spatio2}_grid"].values, sub[f"{Spatio1}_grid"].values] = np.log(sub[col] + adder)
        else:
            im[sub[f"{Spatio2}_grid"].values, sub[f"{Spatio1}_grid"].values] = sub[col]

        my_cmap = matplotlib.cm.get_cmap(cmap)

        if lightgrey_under:
            my_cmap.set_under("lightgrey")

        scat1 = ax.imshow(im, norm=norm, cmap=my_cmap)

        ax.set_title(f"{Temporal1}: {temporal_value}", fontsize=30)

        old_x_ticks = np.arange(0, xlims[1] - xlims[0], xtick_interval)
        new_x_ticks = np.arange(xlims[0], xlims[1], xtick_interval)[: len(old_x_ticks)]
        ax.set_xticks(old_x_ticks, new_x_ticks)

        old_y_ticks = np.arange(0, ylims[1] - ylims[0], ytick_interval)
        new_y_ticks = np.arange(ylims[0], ylims[1], ytick_interval)[::-1][: len(old_y_ticks)]
        ax.set_yticks(old_y_ticks, new_y_ticks)

        if grid:
            plt.grid(alpha=0.5)
        plt.tight_layout()

        return (scat1,)

    # scale the color norm
    #
    if vmax is None:
        if log_scale:
            vmax = np.quantile(np.log(data[col].values + adder), 0.9)
        else:
            vmax = np.quantile(data[col].values, 0.9)

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

    # animate!
    ani = FuncAnimation(
        fig,
        partial(animate, norm=norm, log_scale=log_scale),
        interval=40,
        blit=True,
        repeat=True,
        frames=len(data["Temporal_indexer"].unique()),
    )
    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    plt.close()
    print()
    print("Finish!")


def make_sample_gif_scatter(
    data: pandas.core.frame.DataFrame,
    file_path: str,
    col: str = "abundance",
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    figsize: Tuple[Union[float, int]] = (18, 9),
    xlims: Tuple[Union[float, int]] = (-180, 180),
    ylims: Tuple[Union[float, int]] = (-90, 90),
    grid: bool = True,
    log_scale: bool = False,
    s: float = 0.2,
    dpi: Union[float, int] = 300,
    fps: int = 30,
):
    """make GIF with plt.scatter function

    A function to generate GIF file of spatio-temporal pattern.

    Parameters:
        data:
            Input dataframe
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
            xlim of the figure. In matplotlib style.
        ylims:
            ylim of the figure. In matplotlib style.
        grid:
            Whether to add grids.
        log_scale:
            log transform the target value or not.
        s:
            size of the scatter.
        dpi:
            dpi of the GIF.
        fps:
            speed of GIF playing (frames per second).

    """
    #
    data = data.sort_values(by=Temporal1)
    data["Temporal_indexer"] = LabelEncoder().fit_transform(data[Temporal1])

    fig, ax = plt.subplots(figsize=figsize)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])

    def animate(i, log_scale=log_scale):
        print(i, end=".")
        ax.clear()
        sub = data[data["Temporal_indexer"] == i]
        temporal_value = np.array(sub[Temporal1].values)[0]

        if log_scale:
            sub[col] = np.log(sub[col] + 1)
        else:
            pass

        scat1 = ax.scatter(sub[Spatio1], sub[Spatio2], s=s, c=sub[col], marker="s")

        ax.set_title(f"{Temporal1}: {temporal_value}", fontsize=30)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        plt.tight_layout()
        if grid:
            plt.grid(alpha=0.5)

        return (scat1,)

    # scale the color norm
    if log_scale:
        norm = matplotlib.colors.Normalize(vmin=np.log(data[col].min() + 1), vmax=np.log(data[col].max() + 1))
    else:
        norm = matplotlib.colors.Normalize(vmin=data[col].min(), vmax=data[col].max())

    # for getting the color bar
    scat1 = animate(0)

    cbar = fig.colorbar(scat1[0], norm=norm, shrink=0.5)
    cbar.ax.get_yaxis().labelpad = 15
    if log_scale:
        cbar.ax.set_ylabel(f"log {col}", rotation=270)
    else:
        cbar.ax.set_ylabel(f"{col}", rotation=270)
    plt.tight_layout()
    if grid:
        plt.grid(alpha=0.5)

    # animate!
    ani = FuncAnimation(
        fig, animate, interval=40, blit=True, repeat=True, frames=len(data["Temporal_indexer"].unique())
    )
    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    plt.close()
    print()
    print("Finish!")
