from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def make_sample_gif(
    data: pd.DataFrame,
    file_path: str,
    col: str = "abundance",
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    continental_boundary: bool = True,
    figsize: Tuple[Union[float, int], Union[float, int]] = (18, 9),
    xlims: Tuple[Union[float, int], Union[float, int]] = None,
    ylims: Tuple[Union[float, int], Union[float, int]] = None,
    grid: bool = True,
    lng_size: int = 20,
    lat_size: int = 20,
    xtick_interval: Union[float, int, None] = None,  # used only when continental_boundary=False
    ytick_interval: Union[float, int, None] = None,  # used only when continental_boundary=False
    log_scale: bool = False,
    vmin: Union[float, int] = 0.0001,
    vmax: Union[float, int, None] = None,
    lightgrey_under: bool = True,
    adder: Union[int, float] = 1,
    dpi: Union[float, int] = 300,
    fps: int = 30,
    cmap: str = "plasma",
    verbose: int = 1,
    political_boundary: Optional[str] = None,       # None | "country" | "province" | "both"
    boundary_scale: str = "110m",                    # "110m" | "50m" | "10m"
    boundary_color: str = "black",
    boundary_lw: float = 0.4,
    boundary_alpha: float = 0.7,
    boundary_zorder: int = 2,
    show_major_lakes: bool = True
):
    """
    Create a GIF visualizing spatio-temporal data using imshow, with optional
    Cartopy physical (continental) and political boundaries.

    Args:
        data (pd.DataFrame): Input DataFrame containing spatio-temporal data.
        file_path (str): Output GIF file path.
        col (str): Column name containing the values to visualize (e.g., abundance).
        Spatio1 (str): Column name for the first spatial variable (e.g., longitude).
        Spatio2 (str): Column name for the second spatial variable (e.g., latitude).
        Temporal1 (str): Column name for the temporal variable (e.g., DOY).
        continental_boundary (bool): Whether to display physical continental outlines using Cartopy.
        figsize (Tuple[Union[float, int], Union[float, int]]): Figure size in inches.
        xlims (Tuple[Union[float, int], Union[float, int]]): Longitude limits (min, max).
        ylims (Tuple[Union[float, int], Union[float, int]]): Latitude limits (min, max).
        grid (bool): Whether to draw gridlines on the plot.
        lng_size (int): Number of longitudinal grid cells (spatial resolution).
        lat_size (int): Number of latitudinal grid cells (spatial resolution).
        xtick_interval (Union[float, int, None]): Custom x-axis tick interval (used only if continental_boundary=False).
        ytick_interval (Union[float, int, None]): Custom y-axis tick interval (used only if continental_boundary=False).
        log_scale (bool): Apply logarithmic scaling to the plotted values.
        vmin (Union[float, int]): Minimum value for the colormap normalization.
        vmax (Union[float, int, None]): Maximum value for the colormap normalization (auto-detected if None).
        lightgrey_under (bool): Use light grey color for values below vmin.
        adder (Union[int, float]): Value added before log transformation to avoid log(0).
        dpi (Union[float, int]): Output resolution (dots per inch).
        fps (int): Frames per second for the GIF animation.
        cmap (str): Matplotlib colormap name.
        verbose (int): Verbosity level; 0 = silent, 1 = print progress.
        political_boundary (Optional[str]): Type of political boundaries to overlay.
            Options:
                - None: No political boundaries
                - "country": Show country borders (admin-0)
                - "province": Show state/province boundaries (admin-1)
                - "both": Show both country and province boundaries
        boundary_scale (str): Scale for boundary data ("110m", "50m", or "10m").
        boundary_color (str): Color of physical and political boundaries.
        boundary_lw (float): Line width of boundaries.
        boundary_alpha (float): Transparency (alpha) of boundary lines.
        boundary_zorder (int): Z-order (drawing order) of boundary layers.
        show_major_lakes (bool): Whether to draw outlines of major lakes.

    Returns:
        None. Saves the generated GIF to the specified file path.

    Notes:
        - Spatial binning is performed using `np.digitize` to create a gridded raster.
        - Each frame corresponds to a unique value in the temporal column.
        - The function supports both linear and log-scaled color mapping.
        - Requires `cartopy` for geographic projections and natural features.

    Example:
        >>> make_sample_gif(
        ...     data=df,
        ...     file_path="output.gif",
        ...     col="abundance",
        ...     Spatio1="longitude",
        ...     Spatio2="latitude",
        ...     Temporal1="DOY",
        ...     political_boundary="both",
        ...     log_scale=True
        ... )
    """

    # Sort data by the temporal variable & make frame index
    data = data.sort_values(by=Temporal1)
    data["Temporal_indexer"], _ = pd.factorize(data[Temporal1])
    frames = data["Temporal_indexer"].nunique()

    # Spatial bounds
    if xlims is None:
        xlims = (float(data[Spatio1].min()), float(data[Spatio1].max()))
    if ylims is None:
        ylims = (float(data[Spatio2].min()), float(data[Spatio2].max()))

    # Binning grids (lat reversed so row 0 = max lat; therefore origin='upper')
    lng_grid = np.linspace(xlims[0], xlims[1], lng_size + 1)
    lat_grid = np.linspace(ylims[0], ylims[1], lat_size + 1)[::-1]

    # Color scaling
    if vmax is None:
        vmax = (np.nanmax(np.log(data[col].values + adder))
                if log_scale else np.nanmax(data[col].values))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Colormap
    my_cmap = plt.get_cmap(cmap)
    if lightgrey_under:
        try:
            my_cmap = my_cmap.copy()
        except Exception:
            pass
        my_cmap.set_under("lightgrey")

    # Figure & axes
    if continental_boundary:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_extent([xlims[0], xlims[1], ylims[0], ylims[1]], crs=ccrs.PlateCarree())
    else:
        if political_boundary:
            raise ValueError(
                "political_boundary requires continental_boundary=True."
            )
        if show_major_lakes:
            raise ValueError(
                "show_lakes requires continental_boundary=True."
            )
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # Optional custom ticks only for non-Cartopy axes
        if xtick_interval is not None:
            ax.set_xticks(np.arange(xlims[0], xlims[1] + xtick_interval, xtick_interval))
        if ytick_interval is not None:
            ax.set_yticks(np.arange(ylims[0], ylims[1] + ytick_interval, ytick_interval))
        if grid:
            ax.grid(alpha=0.5)

    # ONE persistent image artist (updated per frame)
    if continental_boundary:
        im = ax.imshow(
            np.full((lat_size, lng_size), np.nan),
            norm=norm,
            cmap=my_cmap,
            extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
            origin="upper",
            transform=ccrs.PlateCarree(),
            animated=False,
            zorder=1,
            interpolation="nearest",
            resample=False
        )

        # Physical outlines
        ax.coastlines(resolution=boundary_scale, linewidth=boundary_lw*2, zorder=boundary_zorder)
        ax.add_feature(
            cfeature.LAND.with_scale(boundary_scale),
            facecolor="none",
            edgecolor=boundary_color,
            linewidth=boundary_lw,
            zorder=boundary_zorder,
        )

        if show_major_lakes:
            ax.add_feature(
                cfeature.LAKES.with_scale(boundary_scale),   # or "110m" for coarser
                facecolor="none",
                edgecolor=boundary_color,
                alpha=boundary_alpha,
                zorder=boundary_zorder,
            )

        # --- NEW: Political boundaries (admin-0 and/or admin-1) ---
        if political_boundary in {"country", "both"}:
            ax.add_feature(
                cfeature.BORDERS.with_scale(boundary_scale),
                edgecolor=boundary_color,
                linewidth=boundary_lw,
                alpha=boundary_alpha,
                zorder=boundary_zorder,
            )

        if political_boundary in {"province", "both"}:
            provinces = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lines",
                scale=boundary_scale,
                facecolor="none",
            )
            ax.add_feature(
                provinces,
                edgecolor=boundary_color,
                linewidth=boundary_lw * 0.9,
                alpha=boundary_alpha,
                zorder=boundary_zorder,
            )

        if grid:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

    else:
        im = ax.imshow(
            np.full((lat_size, lng_size), np.nan),
            norm=norm,
            cmap=my_cmap,
            extent=None,  # pixel coords
            origin="upper",
            animated=False,
            zorder=1,
        )

    # Colorbar & title
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.ax.set_ylabel(f"log({col})" if log_scale else col, rotation=270, labelpad=15)
    title_txt = ax.set_title("", fontsize=30)

    # Animation update: ONLY update raster + title (no ax.clear())
    def animate(i):
        if verbose >= 1:
            print(f"Processing frame {i+1}/{frames}", end="\r")

        sub = data[data["Temporal_indexer"] == i]
        im_data = np.full((lat_size, lng_size), np.nan)

        if not sub.empty:
            # Bin to grid
            g1 = np.digitize(sub[Spatio1].to_numpy(), lng_grid, right=False) - 1
            g1 = np.clip(g1, 0, lng_size - 1).astype(int)
            g2 = np.digitize(sub[Spatio2].to_numpy(), lat_grid, right=False) - 1
            g2 = np.clip(g2, 0, lat_size - 1).astype(int)

            grouped = (
                sub.assign(**{f"{Spatio1}_grid": g1, f"{Spatio2}_grid": g2})
                .groupby([f"{Spatio2}_grid", f"{Spatio1}_grid"])[col]
                .mean()
            )

            if len(grouped) > 0:
                idx0 = grouped.index.get_level_values(0)
                idx1 = grouped.index.get_level_values(1)
                vals = np.log(grouped.values + adder) if log_scale else grouped.values
                im_data[idx0, idx1] = vals

            temporal_value = sub[Temporal1].iloc[0]
            title_txt.set_text(f"{Temporal1}: {temporal_value}")
        else:
            title_txt.set_text("")

        im.set_data(im_data)
        return (im,)

    ani = FuncAnimation(fig, animate, frames=frames, interval=int(1000 / fps), blit=False, repeat=True)
    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    plt.close(fig)
    if verbose >= 1:
        print("\nAnimation saved successfully!")

