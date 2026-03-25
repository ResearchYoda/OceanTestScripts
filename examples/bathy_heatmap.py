"""
bathy_heatmap.py — Load a saved bathymetry grid and display / export a heatmap.

The .npz file produced by bathymetry_map.py contains:
    depth_sum   : (N, N) float64 — accumulated depth values per cell
    depth_count : (N, N) int32   — number of sonar hits per cell
    grid_res    : float64        — metres per cell
    grid_half   : float64        — half-width of the grid in metres

Usage
-----
    # Display interactively
    python bathy_heatmap.py bathy.npz

    # Save to PNG without showing
    python bathy_heatmap.py bathy.npz --out map.png --no-show

    # Choose a different colormap
    python bathy_heatmap.py bathy.npz --cmap plasma

    # Mask cells with fewer than N sonar hits (reduces noise at grid edges)
    python bathy_heatmap.py bathy.npz --min-hits 3

    # Print depth statistics
    python bathy_heatmap.py bathy.npz --stats

    # Show 3-D surface map
    python bathy_heatmap.py bathy.npz --3d

    # Show both heatmap and 3-D surface together
    python bathy_heatmap.py bathy.npz --3d --show-count

    # Save 3-D map to file
    python bathy_heatmap.py bathy.npz --3d --out-3d surface.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def load_bathy(path: str, min_hits: int = 1) -> dict:
    """
    Load a bathymetry grid saved by bathymetry_map.py.

    Parameters
    ----------
    path     : path to the .npz file
    min_hits : cells with fewer sonar returns than this are masked as NaN

    Returns
    -------
    dict with keys:
        depth      : (N, N) float64 — mean depth per cell (NaN = no data)
        count      : (N, N) int32   — hit count per cell
        grid_res   : float — metres per cell
        grid_half  : float — half-extent of the grid in metres
        extent     : [x_min, x_max, y_min, y_max] in metres  (for imshow)
        x_axis     : 1-D array of easting  values (cell centres)
        y_axis     : 1-D array of northing values (cell centres)
    """
    data = np.load(path)

    depth_sum   = data["depth_sum"]
    depth_count = data["depth_count"]
    grid_res    = float(data["grid_res"])
    grid_half   = float(data["grid_half"])

    # Mean depth; mask cells below the hit threshold
    with np.errstate(invalid="ignore"):
        depth = np.where(depth_count >= min_hits,
                         depth_sum / np.where(depth_count > 0, depth_count, 1),
                         np.nan)

    N      = depth.shape[0]
    coords = np.arange(N) * grid_res - grid_half   # cell-centre positions (m)

    return {
        "depth"    : depth,
        "count"    : depth_count,
        "grid_res" : grid_res,
        "grid_half": grid_half,
        "extent"   : [-grid_half, grid_half, -grid_half, grid_half],
        "x_axis"   : coords,
        "y_axis"   : coords,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap renderer
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    bathy: dict,
    *,
    cmap: str    = "viridis_r",
    title: str   = "Bathymetry Heatmap",
    n_contours: int = 8,
    show_contours: bool = True,
    show_count: bool = False,
    out: str | None = None,
    show: bool  = True,
) -> plt.Figure:
    """
    Render a publication-quality heatmap of the bathymetry grid.

    Parameters
    ----------
    bathy         : dict returned by load_bathy()
    cmap          : matplotlib colormap name (default: viridis_r — deep=dark)
    title         : figure title
    n_contours    : number of depth contour lines overlaid on the heatmap
    show_contours : draw iso-depth contour lines
    show_count    : add a second panel showing sonar hit-count per cell
    out           : if given, save the figure to this path (PNG/PDF/SVG…)
    show          : call plt.show() at the end

    Returns
    -------
    matplotlib Figure
    """
    depth = bathy["depth"]
    count = bathy["count"]
    ext   = bathy["extent"]   # [x_min, x_max, y_min, y_max]

    valid = ~np.isnan(depth)
    if not valid.any():
        raise ValueError("No valid depth data found in the grid.")

    d_min = depth[valid].min()
    d_max = depth[valid].max()

    n_panels = 2 if show_count else 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(7 * n_panels, 6),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: depth heatmap ────────────────────────────────────────────────
    ax = axes[0]

    # imshow: rows = Y (northing), cols = X (easting)
    # origin="lower" puts south at the bottom (conventional map orientation)
    im = ax.imshow(
        depth,
        origin="lower",
        extent=ext,
        cmap=cmap,
        vmin=d_min,
        vmax=d_max,
        interpolation="nearest",
        aspect="equal",
    )

    # Contour lines at evenly spaced depth levels
    if show_contours:
        levels  = np.linspace(d_min, d_max, n_contours + 2)[1:-1]
        X, Y    = np.meshgrid(bathy["x_axis"], bathy["y_axis"])
        depth_f = np.where(valid, depth, np.nan)
        cs = ax.contour(X, Y, depth_f,
                        levels=levels, colors="white",
                        linewidths=0.6, alpha=0.55)
        ax.clabel(cs, fmt="%.1f m", fontsize=7, inline=True)

    # Colourbar
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.05)
    cbar    = fig.colorbar(im, cax=cax)
    cbar.set_label("Depth (m)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax.set_xlabel("Easting (m)",  fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4, color="white")

    # Coverage box annotation
    cov_pct = 100.0 * valid.sum() / depth.size
    ax.text(0.02, 0.98,
            f"Coverage: {cov_pct:.1f}%\n"
            f"Depth:  {d_min:.2f} – {d_max:.2f} m\n"
            f"Res:    {bathy['grid_res']} m/cell",
            transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))

    # ── Panel 2: hit-count map (optional) ─────────────────────────────────────
    if show_count:
        ax2   = axes[1]
        count_f = np.where(count > 0, count, np.nan)
        im2   = ax2.imshow(
            count_f,
            origin="lower",
            extent=ext,
            cmap="hot",
            interpolation="nearest",
            aspect="equal",
        )
        divider2 = make_axes_locatable(ax2)
        cax2     = divider2.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(im2, cax=cax2, label="Sonar hits per cell")

        ax2.set_xlabel("Easting (m)",  fontsize=10)
        ax2.set_ylabel("Northing (m)", fontsize=10)
        ax2.set_title("Sonar Coverage (hit count)", fontsize=12)
        ax2.grid(True, linestyle="--", linewidth=0.3, alpha=0.3, color="grey")

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[heatmap] Saved to {out}")

    if show:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3-D surface map
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d(
    bathy: dict,
    *,
    cmap: str            = "terrain",
    title: str           = "Bathymetry 3D Surface",
    downsample: int      = 4,
    vertical_exag: float = 3.0,
    show_contours: bool  = True,
    n_contours: int      = 8,
    out: str | None      = None,
    elev: float          = 30.0,
    azim: float          = -60.0,
) -> plt.Figure:
    """
    Render the bathymetry grid as a fully interactive 3-D surface.

    The window stays open and is mouse-rotatable.  Moving the cursor over
    the surface highlights the nearest point and shows a live readout of
    its (Easting, Northing, Depth).  Left-click pins/unpins the annotation.

    Depth is inverted to elevation so hills appear as bumps and trenches
    as holes.  Unvisited cells are shown in dark grey.

    Parameters
    ----------
    bathy          : dict returned by load_bathy()
    cmap           : colormap  (default 'terrain': green hills, blue deeps)
    title          : figure title
    downsample     : stride when sampling the grid  (default 4 → 100×100)
    vertical_exag  : depth multiplier for visual relief  (default 3×)
    show_contours  : project iso-depth contours onto the floor plane
    n_contours     : number of contour levels
    out            : save figure to this path before showing  (PNG/PDF/SVG)
    elev / azim    : initial camera angles in degrees

    Returns
    -------
    matplotlib Figure  (blocking — plt.show() is called internally)
    """
    depth = bathy["depth"]
    valid = ~np.isnan(depth)
    if not valid.any():
        raise ValueError("No valid depth data found in the grid.")

    d_min = float(depth[valid].min())
    d_max = float(depth[valid].max())

    # ── Downsample grid ───────────────────────────────────────────────────────
    s    = max(1, downsample)
    x_ds = bathy["x_axis"][::s]          # (nx,)
    y_ds = bathy["y_axis"][::s]          # (ny,)
    z_ds = depth[::s, ::s]               # (ny, nx), may contain NaN

    X, Y     = np.meshgrid(x_ds, y_ds)   # both (ny, nx)
    nan_mask = np.isnan(z_ds)

    # NaN → shallowest depth  →  flat grey "ceiling" for unvisited cells
    z_fill  = np.where(~nan_mask, z_ds, d_max)
    # Elevation: invert depth and apply exaggeration
    Z_elev  = -z_fill * vertical_exag    # (ny, nx)

    floor_z = Z_elev.min() - max(1.0, abs(Z_elev.min()) * 0.06)

    # ── Colour mapping ────────────────────────────────────────────────────────
    norm         = Normalize(vmin=d_min, vmax=d_max)
    cmap_obj     = plt.get_cmap(cmap)
    face_colours = cmap_obj(1.0 - norm(z_fill))   # shallow = bright
    face_colours[nan_mask] = (0.22, 0.22, 0.22, 1.0)

    # ── Figure & axes ─────────────────────────────────────────────────────────
    fig = plt.figure("Bathymetry 3D", figsize=(11, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#16213e")
    ax.view_init(elev=elev, azim=azim)

    # Surface
    ax.plot_surface(
        X, Y, Z_elev,
        facecolors=face_colours,
        linewidth=0,
        antialiased=True,
        shade=True,
        zorder=1,
    )

    # ── Floor contours ────────────────────────────────────────────────────────
    if show_contours and (d_max - d_min) > 0.01:
        levels_d   = np.linspace(d_min, d_max, n_contours + 2)[1:-1]
        levels_elv = sorted(-lv * vertical_exag for lv in levels_d)
        Z_ct       = np.where(nan_mask, np.nan, Z_elev)
        ax.contourf(X, Y, Z_ct, levels=levels_elv,
                    zdir="z", offset=floor_z, cmap=cmap, alpha=0.40, zorder=0)
        cs = ax.contour(X, Y, Z_ct, levels=levels_elv,
                        zdir="z", offset=floor_z,
                        colors="white", linewidths=0.6, alpha=0.55, zorder=0)
        ax.clabel(cs, fmt=lambda v: f"{-v/vertical_exag:.1f}m",
                  fontsize=6, inline=True)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.50, pad=0.10, aspect=18,
                        location="right")
    cbar.set_label("Depth (m)", fontsize=10, color="white")
    cbar.ax.invert_yaxis()
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(colors="white", labelsize=8)

    # ── Axis styling ──────────────────────────────────────────────────────────
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.label.set_color("white")
        spine.set_tick_params(colors="white", labelsize=7)

    ax.set_xlabel("Easting (m)",  fontsize=9, labelpad=8, color="white")
    ax.set_ylabel("Northing (m)", fontsize=9, labelpad=8, color="white")
    ax.set_zlabel(f"Depth ×{vertical_exag:.0f}", fontsize=9,
                  labelpad=8, color="white")
    ax.set_zlim(floor_z, Z_elev.max() + 0.5)

    # Z-tick labels → real depth in metres
    ax.set_zticks(np.linspace(floor_z, Z_elev.max(), 6))
    ax.set_zticklabels(
        [f"{-t / vertical_exag:.1f}" for t in np.linspace(floor_z, Z_elev.max(), 6)],
        fontsize=7)

    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=14)

    # Stats box (top-left)
    cov_pct = 100.0 * valid.sum() / depth.size
    ax.text2D(0.01, 0.98,
              f"Coverage : {cov_pct:.1f}%\n"
              f"Depth    : {d_min:.2f} – {d_max:.2f} m\n"
              f"V. exag  : ×{vertical_exag:.1f}  |  res {bathy['grid_res']} m",
              transform=ax.transAxes, va="top", ha="left",
              fontsize=8, color="white",
              bbox=dict(boxstyle="round,pad=0.4",
                        facecolor="#0f3460", alpha=0.80, edgecolor="none"))

    # ── Interactive cursor ────────────────────────────────────────────────────
    # Flatten all *valid* downsampled points for nearest-point lookup
    _xf = X[~nan_mask].ravel()
    _yf = Y[~nan_mask].ravel()
    _zf = Z_elev[~nan_mask].ravel()         # exaggerated elevation
    _df = z_ds[~nan_mask].ravel()           # real depth (m)

    # Cursor marker: a red dot that moves with the mouse
    _cursor_dot, = ax.plot([], [], [], "o",
                           color="#ff4757", markersize=8, zorder=10,
                           markeredgecolor="white", markeredgewidth=0.8)

    # Vertical drop-line from the dot to the floor
    _cursor_line, = ax.plot([], [], [], "--",
                            color="#ff4757", linewidth=0.9,
                            alpha=0.6, zorder=9)

    # Text readout pinned to the bottom of the axes
    _readout = ax.text2D(
        0.50, 0.02,
        "Move cursor over the surface",
        transform=ax.transAxes,
        ha="center", va="bottom", fontsize=9, color="#f1f1f1",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#0f3460", alpha=0.85, edgecolor="none"),
    )

    # State: True = user has clicked to pin the annotation
    _pinned = [False]

    def _nearest_point(event):
        """Return index of the grid point whose 2-D screen projection is
        closest to the mouse cursor, or None if the mouse is outside the axes."""
        if event.inaxes is not ax:
            return None
        try:
            proj = ax.get_proj()
            u, v, _ = proj3d.proj_transform(_xf, _yf, _zf, proj)
            # u, v are in normalised display coords; convert to pixels
            pts2d = ax.transData.transform(np.column_stack([u, v]))
            dist  = np.hypot(pts2d[:, 0] - event.x, pts2d[:, 1] - event.y)
            return int(dist.argmin())
        except Exception:
            return None

    def _update_cursor(idx):
        ex, ny, ze, de = _xf[idx], _yf[idx], _zf[idx], _df[idx]
        _cursor_dot.set_data_3d([ex], [ny], [ze])
        _cursor_line.set_data_3d([ex, ex], [ny, ny], [ze, floor_z])
        _readout.set_text(
            f"Easting: {ex:+.1f} m    Northing: {ny:+.1f} m    Depth: {de:.2f} m")
        fig.canvas.draw_idle()

    def _on_move(event):
        if _pinned[0]:
            return
        idx = _nearest_point(event)
        if idx is not None:
            _update_cursor(idx)

    def _on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        idx = _nearest_point(event)
        if idx is None:
            return
        if _pinned[0]:
            # Second click: unpin
            _pinned[0] = False
            _readout.set_bbox(dict(boxstyle="round,pad=0.4",
                                   facecolor="#0f3460", alpha=0.85,
                                   edgecolor="none"))
        else:
            # First click: pin
            _pinned[0] = True
            _update_cursor(idx)
            _readout.set_bbox(dict(boxstyle="round,pad=0.4",
                                   facecolor="#c0392b", alpha=0.90,
                                   edgecolor="white", linewidth=0.8))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    fig.canvas.mpl_connect("button_press_event",  _on_click)

    # ── Save & show ───────────────────────────────────────────────────────────
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[3d] Saved to {out}")

    plt.show(block=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(bathy: dict) -> None:
    """Print a summary of depth statistics to stdout."""
    depth = bathy["depth"]
    count = bathy["count"]
    valid = ~np.isnan(depth)

    if not valid.any():
        print("No valid depth data.")
        return

    d      = depth[valid]
    N      = depth.shape[0]
    res    = bathy["grid_res"]
    half   = bathy["grid_half"]

    print("=== Bathymetry Statistics ===")
    print(f"  File grid size   : {N} × {N} cells")
    print(f"  Cell resolution  : {res} m")
    print(f"  Spatial extent   : ±{half} m  ({2*half:.0f} × {2*half:.0f} m)")
    print(f"  Cells with data  : {valid.sum():,} / {depth.size:,}  "
          f"({100*valid.sum()/depth.size:.1f}%)")
    print(f"  Total sonar hits : {count.sum():,}")
    print(f"  Depth  min       : {d.min():.3f} m")
    print(f"  Depth  max       : {d.max():.3f} m")
    print(f"  Depth  mean      : {d.mean():.3f} m")
    print(f"  Depth  std dev   : {d.std():.3f} m")
    print(f"  Depth  median    : {np.median(d):.3f} m")

    # Percentile table
    pcts = [5, 25, 50, 75, 95]
    vals = np.percentile(d, pcts)
    print("  Depth percentiles:")
    for p, v in zip(pcts, vals):
        print(f"    {p:3d}th  →  {v:.3f} m")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load a bathy.npz file and render a depth heatmap.")
    parser.add_argument("npz",
                        help="Path to the .npz bathymetry file")
    parser.add_argument("--cmap",      default="viridis_r",
                        help="Matplotlib colormap (default: viridis_r)")
    parser.add_argument("--out",       default=None,
                        help="Save figure to this file (e.g. map.png)")
    parser.add_argument("--no-show",   action="store_true",
                        help="Do not open an interactive window")
    parser.add_argument("--no-contours", action="store_true",
                        help="Omit iso-depth contour lines")
    parser.add_argument("--show-count", action="store_true",
                        help="Add a second panel showing sonar hit-count")
    parser.add_argument("--min-hits",  type=int, default=1,
                        help="Mask cells with fewer hits than this (default: 1)")
    parser.add_argument("--stats",       action="store_true",
                        help="Print depth statistics to stdout")
    parser.add_argument("--title",       default="Bathymetry Heatmap",
                        help="Figure title")
    # 3-D surface options
    parser.add_argument("--3d",          action="store_true", dest="show_3d",
                        help="Show a 3-D surface map")
    parser.add_argument("--no-2d",       action="store_true",
                        help="Skip the 2-D heatmap (only show 3-D)")
    parser.add_argument("--out-3d",      default=None, dest="out_3d",
                        help="Save 3-D figure to this file (e.g. surface.png)")
    parser.add_argument("--cmap-3d",     default="terrain", dest="cmap_3d",
                        help="Colormap for the 3-D surface (default: terrain)")
    parser.add_argument("--vexag",       type=float, default=3.0,
                        help="Vertical exaggeration factor for 3-D (default: 3.0)")
    parser.add_argument("--downsample",  type=int, default=4,
                        help="Downsample factor for 3-D surface (default: 4)")
    parser.add_argument("--elev",        type=float, default=30.0,
                        help="Camera elevation angle for 3-D view (default: 30)")
    parser.add_argument("--azim",        type=float, default=-60.0,
                        help="Camera azimuth angle for 3-D view (default: -60)")
    args = parser.parse_args()

    bathy = load_bathy(args.npz, min_hits=args.min_hits)

    if args.stats:
        print_stats(bathy)

    if not args.no_2d:
        plot_heatmap(
            bathy,
            cmap          = args.cmap,
            title         = args.title,
            show_contours = not args.no_contours,
            show_count    = args.show_count,
            out           = args.out,
            show          = not args.no_show,
        )

    if args.show_3d:
        plot_3d(
            bathy,
            cmap          = args.cmap_3d,
            title         = args.title.replace("Heatmap", "3D Surface"),
            downsample    = args.downsample,
            vertical_exag = args.vexag,
            out           = args.out_3d,
            elev          = args.elev,
            azim          = args.azim,
        )


if __name__ == "__main__":
    main()
