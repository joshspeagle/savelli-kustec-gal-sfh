#############################################################
#                                                           #
#           Common plotting utilities for SFH analysis     #
#                                                           #
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import (
    Normalize,
    LogNorm,
)
from matplotlib import cm

from .analysis import sim_name, sim_name_short, times

# =============================================================================
# Color and Display Utilities
# =============================================================================


def is_dark(color):
    """
    Determine if color is too dark for black text.

    Parameters
    ----------
    color : tuple
        RGBA tuple (values in 0–1)

    Returns
    -------
    bool
        True if color is too dark for black text.
    """
    r, g, b = color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 0.4


# =============================================================================
# Grid Drawing Functions
# =============================================================================


def draw_battleship_grid(
    ax, labelleft=True, labelright=True, labelbottom=True, labeltop=True
):
    """
    Draw the 10x10 battleship grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    labelleft : bool, default=True
        Show left labels
    labelright : bool, default=True
        Show right labels
    labelbottom : bool, default=True
        Show bottom labels
    labeltop : bool, default=True
        Show top labels
    """

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # grid
    for i in range(11):
        ax.axhline(i, color="dimgrey", linestyle="--", linewidth=1)
        ax.axvline(i, color="dimgrey", linestyle="--", linewidth=1)

    ax.tick_params(
        labelbottom=labelbottom,
        labeltop=labeltop,
        labelleft=labelleft,
        labelright=labelright,
        left=False,
        right=False,
        top=False,
        bottom=False,
        labelsize=20,
    )
    ax.set_xticks(
        np.linspace(0, 9, 10) + 0.5,
        labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )
    ax.set_yticks(
        np.linspace(0, 9, 10) + 0.5,
        labels=["$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$", "$10$"][
            ::-1
        ],
    )

    ax.set_aspect("equal")


def draw_grid_A(xlabel=None, ylabel=None, xlim=None, ylim=None, umap=False):
    """
    Draw grid type A, for displaying plot for each individual sim + big plot for all sims combined.

    Parameters
    ----------
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    umap : bool, default=False
        Use UMAP-specific formatting

    Returns
    -------
    axes : dict
        Dictionary of axes objects keyed by simulation names and 'large'
    """
    if umap:
        fig = plt.figure(figsize=(15, 15))
    else:
        fig = plt.figure(figsize=(10, 10))

    gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0, hspace=0)
    axes = dict.fromkeys(sim_name)

    # draw small grids
    for row in range(3):
        for col in range(4):
            if row in [1, 2] and col in [2, 3]:
                continue
            ax = fig.add_subplot(gs[row, col])
            if row == 0 and col == 3:
                ax.axis("off")
            else:
                # format axes
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                if umap:
                    # Draw battleship grid
                    labelleft = True
                    labelright = True
                    labeltop = True
                    labelbottom = True
                    if col not in [0, 3]:
                        labelleft = False
                    if row == 0:
                        labelbottom = False
                        labeltop = True
                        if col == 2:
                            labelleft = False
                            labelright = True
                    elif row == 2:
                        labelbottom = True
                        labeltop = False
                    else:
                        labeltop = False
                        labelbottom = False
                    draw_battleship_grid(
                        ax,
                        labelleft=labelleft,
                        labelright=labelright,
                        labeltop=labeltop,
                        labelbottom=labelbottom,
                    )
                else:
                    # Enable ticks on all sides
                    ax.tick_params(
                        which="both",
                        top=True,
                        bottom=True,
                        left=True,
                        right=True,
                        direction="in",
                        labelsize=16,
                    )

                    # Hide interior tick labels
                    if col not in [0, 3]:
                        ax.tick_params(labelleft=False)
                    elif ylabel:
                        ax.set_ylabel(ylabel, fontsize=20)
                    if row != 2:
                        ax.tick_params(labelbottom=False)
                    elif xlabel:
                        ax.set_xlabel(xlabel, fontsize=20)

            # name axes
            if (row == 0) & (col == 0):
                axes["EAGLE"] = ax
            elif (row == 0) & (col == 1):
                axes["Illustris"] = ax
            elif (row == 0) & (col == 2):
                axes["IllustrisTNG"] = ax
            elif (row == 1) & (col == 0):
                axes["Mufasa"] = ax
            elif (row == 1) & (col == 1):
                axes["Simba"] = ax
            elif (row == 2) & (col == 0):
                axes["SC-SAM"] = ax
            elif (row == 2) & (col == 1):
                axes["UniverseMachine"] = ax

    # draw large grid
    ax_large = fig.add_subplot(gs[1:3, 2:4])

    if umap:
        # Draw battleship grid
        draw_battleship_grid(
            ax_large, labelbottom=True, labeltop=False, labelleft=False, labelright=True
        )
    else:
        ax_large.tick_params(
            which="both",
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelleft=False,
            labelright=True,
            direction="in",
            labelsize=16,
        )
    if xlim:
        ax_large.set_xlim(xlim)
    if ylim:
        ax_large.set_ylim(ylim)
    if xlabel:
        ax_large.set_xlabel(xlabel, fontsize=20)
    if ylabel:
        ax_large.set_ylabel(ylabel, fontsize=20)
        ax_large.yaxis.set_label_position("right")

    axes["large"] = ax_large

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return axes


def draw_grid_B(cmap=None, vmin=None, vmax=None, lognorm=False):
    """
    Draw grid type B, for displaying "point" plot, cell average value plot, and avg SFH plots.

    Parameters
    ----------
    cmap : colormap, optional
        Colormap for plotting
    vmin : float, optional
        Minimum value in "point" data, to normalize colorbar along all 3 plots
    vmax : float, optional
        Maximum value in "point" data, to normalize colorbar along all 3 plots
    lognorm : bool, default=False
        Use log normalization for colorbar

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axs : list
        List of axes objects
    norm : matplotlib.colors.Normalize
        Color normalization object
    """
    # Color normalization
    if lognorm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Main figure and subfigures
    fig = plt.figure(figsize=(12, 12))

    # --- Axes layout using manual positioning for full square control ---

    # Two square axes on top
    ax1 = fig.add_axes([0.05, 0.72, 0.4, 0.4])  # left, bottom, width, height
    ax2 = fig.add_axes([0.45, 0.72, 0.4, 0.4])

    # One big square axis at the bottom
    ax3 = fig.add_axes([0.2, 0.14, 0.5, 0.5])
    cax = fig.add_axes([0.75, 0.14, 0.03, 0.5])  # Colorbar next to it

    axs = [ax1, ax2, ax3, cax]

    draw_battleship_grid(ax1, labelright=False)
    draw_battleship_grid(ax2, labelleft=False)
    draw_battleship_grid(ax3)

    return fig, axs, norm


def draw_grid_C(cmap=None, vmin=None, vmax=None, lognorm=False, cbar=True):
    """
    Draw grid type C, for displaying three battleship plots.

    Parameters
    ----------
    cmap : colormap, optional
        Colormap for plotting
    vmin : float, optional
        Minimum value in "point" data, to normalize colorbar along all 3 plots
    vmax : float, optional
        Maximum value in "point" data, to normalize colorbar along all 3 plots
    lognorm : bool, default=False
        Use log normalization for colorbar
    cbar : bool, default=True
        Show colorbar

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axs : list
        List of axes objects
    norm : matplotlib.colors.Normalize
        Color normalization object
    """

    # Color normalization
    if lognorm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Main figure and subfigures
    fig = plt.figure(figsize=(15, 15))

    # --- Axes layout using manual positioning for full square control ---

    # draw axes
    ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.3])  # left, bottom, width, height
    ax2 = fig.add_axes([0.35, 0.1, 0.3, 0.3])
    ax3 = fig.add_axes([0.65, 0.1, 0.3, 0.3])
    if cbar:
        cax = fig.add_axes([0.98, 0.1, 0.03, 0.3])  # Colorbar axis
    else:
        cax = None

    axs = [ax1, ax2, ax3, cax]

    draw_battleship_grid(ax1, labelright=False)
    draw_battleship_grid(ax2, labelleft=False, labelright=False)
    draw_battleship_grid(ax3, labelleft=False)

    return fig, axs, norm


def draw_grid_D(
    cmap=None,
    vmin=None,
    vmax=None,
    lognorm=False,
    cbar=True,
):
    """
    Draw grid type D, for displaying six battleship plots in 2 rows with single spanning colorbar.

    Parameters
    ----------
    cmap : colormap, optional
        Colormap for all plots
    vmin : float, optional
        Minimum value for colorbar
    vmax : float, optional
        Maximum value for colorbar
    lognorm : bool, default=False
        Use log normalization for colorbar
    cbar : bool, default=True
        Show colorbar

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axs : list
        List of axes objects
    norm : matplotlib.colors.Normalize
        Color normalization object
    """

    # Color normalization
    if lognorm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Main figure and subfigures
    fig = plt.figure(figsize=(15, 15))

    # --- Axes layout using manual positioning for full square control ---

    # draw axes
    ax1 = fig.add_axes([0.05, 0.4, 0.3, 0.3])  # left, bottom, width, height
    ax2 = fig.add_axes([0.35, 0.4, 0.3, 0.3])
    ax3 = fig.add_axes([0.65, 0.4, 0.3, 0.3])
    ax4 = fig.add_axes([0.05, 0.1, 0.3, 0.3])  # left, bottom, width, height
    ax5 = fig.add_axes([0.35, 0.1, 0.3, 0.3])
    ax6 = fig.add_axes([0.65, 0.1, 0.3, 0.3])
    if cbar:
        cax = fig.add_axes([0.98, 0.1, 0.03, 0.6])  # Spanning colorbar axis
    else:
        cax = None

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, cax]

    draw_battleship_grid(ax1, labelright=False, labelbottom=False)
    draw_battleship_grid(ax2, labelleft=False, labelright=False, labelbottom=False)
    draw_battleship_grid(ax3, labelleft=False, labelbottom=False)
    draw_battleship_grid(ax4, labelright=False, labeltop=False)
    draw_battleship_grid(ax5, labelleft=False, labelright=False, labeltop=False)
    draw_battleship_grid(ax6, labelleft=False, labeltop=False)

    return fig, axs, norm


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_points(
    x,
    fig,
    ax,
    cax,
    cmap,
    norm,
    extend="neither",
    label=None,
    y=None,
    z=None,
    s=1,
    alpha=1,
    imshow=False,
    contourf=False,
):
    """
    Draw "point" plot in battleship grid.

    Parameters
    ----------
    x : array_like
        Data to plot
    fig : matplotlib.figure.Figure
        Figure to plot on
    ax : matplotlib.axes.Axes
        Axes to plot on
    cax : matplotlib.axes.Axes
        Location of colorbar
    cmap : colormap
        Colormap
    norm : matplotlib.colors.Normalize
        Color normalization
    extend : str, default='neither'
        Extend colorbar ('neither', 'both', 'min', 'max')
    label : str, optional
        Colorbar label
    y : array_like, optional
        Second dimension of data, if needed
    z : array_like, optional
        Third dimension of data, if needed (for color bar)
    s : float, default=1
        Marker size for scatter
    alpha : float, default=1
        Alpha transparency
    imshow : bool, default=False
        Plot as imshow object (defaults to scatter plot)
    contourf : bool, default=False
        Plot contours (defaults to scatter plot)
    """

    if imshow:
        im = ax.imshow(
            x.T,  # transpose to match x/y orientation
            origin="lower",  # put [0,0] in bottom-left
            extent=[0, 10, 0, 10],  # map array coords to data coords
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
    elif contourf:
        im = ax.contourf(x.T, cmap=cmap, norm=norm, extent=[0, 10, 0, 10])
    else:
        im = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=s, alpha=alpha)

    # colour bar
    if cax is not None:
        cbar = fig.colorbar(im, cax=cax, extend=extend)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label, fontsize=20)


def plot_averages(
    avg,
    avg_sfh,
    fig,
    ax1,
    ax2,
    umaps,
    cmap,
    norm,
    cax=None,
    ftype="f",
    plot_text=True,
    plot_cells=True,
    label=None,
):
    """
    Draw "average" plot in battleship grid.

    Parameters
    ----------
    avg : array_like
        Average per cell array
    avg_sfh : array_like
        Average SFH array (percentiles)
    fig : matplotlib.figure.Figure
        Figure to plot on
    ax1 : matplotlib.axes.Axes
        Location of axis for cell avg plot
    ax2 : matplotlib.axes.Axes
        Location of average for SFH plot
    umaps : array_like
        Array of umap positions sorted into 10x10 battleship gridcells
    cmap : colormap
        Colormap
    norm : matplotlib.colors.Normalize
        Colorbar normalization
    cax : matplotlib.axes.Axes, optional
        Colorbar axis location
    ftype : str, default='f'
        Number format type ('f', 'e')
    plot_text : bool, default=True
        Whether to plot text values in ax1
    plot_cells : bool, default=True
        Whether to plot cell values in ax1
    label : str, optional
        Label for colorbar
    """
    # avg plots
    for i in range(10):
        for j in range(10):
            x0 = i
            y0 = j
            if np.isnan(avg[i, j]):
                colour = "w"
            else:
                colour = cmap(norm(avg[i, j]))

            # plot average cell value
            count = len(umaps[i, j])
            if count >= 100:
                if plot_cells:
                    ax1.fill(
                        [x0, x0 + 1, x0 + 1, x0], [y0, y0, y0 + 1, y0 + 1], fc=colour
                    )
                if plot_text and ftype == "e":
                    s = f"{avg[i,j]:.0e}"
                    fontsize = 14
                    if s[-3] == "+":
                        s = s[:-3] + s[-1] if s[-2] == "0" else s[:-3] + s[-2:]
                        if is_dark(colour):
                            ax1.text(
                                i + 0.5,
                                j + 0.5,
                                s,
                                va="center",
                                ha="center",
                                c="w",
                                fontsize=fontsize,
                            )
                        else:
                            ax1.text(
                                i + 0.5,
                                j + 0.5,
                                s,
                                va="center",
                                ha="center",
                                c="k",
                                fontsize=fontsize,
                            )
                    else:
                        s = s[:-2] + s[-1] if s[-2] == "0" else s[:-2] + s[-2:]
                        if is_dark(colour):
                            ax1.text(
                                i + 0.5,
                                j + 0.5,
                                s,
                                va="center",
                                ha="center",
                                c="w",
                                fontsize=fontsize - 2,
                            )
                        else:
                            ax1.text(
                                i + 0.5,
                                j + 0.5,
                                s,
                                va="center",
                                ha="center",
                                c="k",
                                fontsize=fontsize - 2,
                            )
                elif plot_text and ftype == "f":
                    digits = -int(np.floor(np.log10(np.abs(avg[i, j])))) + (3 - 1)
                    if is_dark(colour):
                        ax1.text(
                            i + 0.5,
                            j + 0.5,
                            f"{avg[i,j]:.{digits}f}".lstrip("0"),
                            va="center",
                            ha="center",
                            c="w",
                            fontsize=16,
                        )
                    else:
                        ax1.text(
                            i + 0.5,
                            j + 0.5,
                            f"{(avg[i,j]):.{digits}f}".lstrip("0"),
                            va="center",
                            ha="center",
                            c="k",
                            fontsize=16,
                        )
                elif plot_text:
                    print("I only know format types f, e")

                # plot avg SFHs
                ins = ax2.inset_axes([i * 0.1, j * 0.1, 0.1, 0.1])
                ins.axis("off")
                ins.semilogy(times, avg_sfh[i, j][1], c=colour)
                ins.fill_between(
                    times, avg_sfh[i, j][0], avg_sfh[i, j][2], color=colour, alpha=0.3
                )
                ins.set_ylim(10**-12.9, 10**-9.1)  # same as fig 2

    # Add colorbar if cax is provided
    if cax is not None:
        import matplotlib as mpl

        im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        if label:
            cbar.set_label(label, fontsize=20)


def plot_cm(cm, fig, ax, cmap, title=None, left=True, zero=False):
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm : array_like
        Confusion matrix (7x7 array)
    fig : matplotlib.figure.Figure
        Figure to plot on
    ax : matplotlib.axes.Axes
        Axes
    cmap : colormap
        Colormap
    title : str, optional
        Title
    left : bool, default=True
        Plot on left or right side?
    zero : bool, default=False
        Hatch out zeros?
    """
    norm = Normalize(0, 1)

    for i in range(7):
        for j in range(7):
            x0 = i
            y0 = -j
            value = cm[j, i]

            # colour box
            if (value == 0.0) and zero:
                colour = "grey"
                hatch = "//"
                alpha = 0.6
            else:
                colour = cmap(norm(value))
                hatch = None
                alpha = 1
            ax.fill(
                [x0, x0 - 1, x0 - 1, x0],
                [y0, y0, y0 + 1, y0 + 1],
                fc=colour,
                hatch=hatch,
                alpha=alpha,
            )
            plt.rcParams["hatch.color"] = "grey"

            # write value
            if (value == 0) & ~zero:
                ax.text(
                    i - 0.5,
                    -j + 0.5,
                    "0",
                    va="center",
                    ha="center",
                    c="k" if zero else "w",
                    fontsize=18,
                )
            elif value > 0:
                if is_dark(colour):
                    ax.text(
                        i - 0.5,
                        -j + 0.5,
                        f"{value:.4f}".lstrip("0"),
                        va="center",
                        ha="center",
                        c="w",
                        fontsize=20,
                    )
                else:
                    ax.text(
                        i - 0.5,
                        -j + 0.5,
                        f"{value:.4f}".lstrip("0"),
                        va="center",
                        ha="center",
                        c="k",
                        fontsize=20,
                    )

            ax.set_xlim(-1, 6)
            ax.set_ylim(-6, 1)
            ax.tick_params(
                left=left,
                bottom=False,
                right=not (left),
                labelleft=left,
                labelright=not (left),
                labelbottom=False,
                labeltop=True,
            )
            ax.set_xticks(
                np.arange(-0.5, 6, 1), labels=sim_name_short, rotation=90, fontsize=20
            )
            ax.set_yticks(
                np.arange(-5.5, 1, 1), labels=sim_name_short[::-1], fontsize=20
            )
            ax.set_xlabel("Predicted Sim", fontsize=20)
            ax.set_ylabel("True Sim", fontsize=20)
            ax.xaxis.set_label_position("top")
            if not (left):
                ax.yaxis.set_label_position("right")
            ax.set_title(title, fontsize=26)


def plot_acc(acc, fig, ax, cmap, left=True, zero=False):
    """
    Plot accuracy in UMAP battleship grid.

    Parameters
    ----------
    acc : array_like
        Accuracy values for each battleship gridcell (10x10 array)
    fig : matplotlib.figure.Figure
        Figure to plot on
    ax : matplotlib.axes.Axes
        Axes
    cmap : colormap
        Colormap
    left : bool, default=True
        Plot on left or right side?
    zero : bool, default=False
        Hatch out zeros?
    """
    norm = Normalize(0, 1)

    draw_battleship_grid(ax, labelright=not (left), labelleft=left)

    for i in range(10):
        for j in range(10):
            x0 = i
            y0 = j
            value = acc[i, j]

            # colour box
            if ~np.isfinite(value):
                colour = "k"
                hatch = None
                alpha = 1.0
            elif (value == 0.0) & zero:
                colour = "grey"
                hatch = "//"
                alpha = 0.6
            else:
                colour = cmap(norm(value))
                hatch = None
                alpha = 1
            ax.fill(
                [x0, x0 + 1, x0 + 1, x0],
                [y0, y0, y0 + 1, y0 + 1],
                fc=colour,
                hatch=hatch,
                alpha=alpha,
            )
            plt.rcParams["hatch.color"] = "grey"

            # write value
            if np.isfinite(value):
                if (value == 0) & ~zero:
                    ax.text(
                        i + 0.5,
                        j + 0.5,
                        "0",
                        va="center",
                        ha="center",
                        c="k" if zero else "w",
                        fontsize=20,
                    )
                elif (value < 1) & (value > 0):
                    if is_dark(colour):
                        ax.text(
                            i + 0.5,
                            j + 0.5,
                            f"{value:.3f}".lstrip("0"),
                            va="center",
                            ha="center",
                            c="w",
                            fontsize=20,
                        )
                    else:
                        ax.text(
                            i + 0.5,
                            j + 0.5,
                            f"{value:.3f}".lstrip("0"),
                            va="center",
                            ha="center",
                            c="k",
                            fontsize=20,
                        )
                else:
                    ax.text(
                        i + 0.5,
                        j + 0.5,
                        f"{value:.2f}",
                        va="center",
                        ha="center",
                        c="k",
                        fontsize=20,
                    )


def plot_sop_contours(p_s_x, sim_name, colors, fig, ax, cax=None, label=None):
    """
    Plot simulation origin probability contours.

    Parameters
    ----------
    p_s_x : dict
        Dictionary of simulation origin probabilities for each simulation
    sim_name : list
        List of simulation names
    colors : dict
        Dictionary of colors for each simulation
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object for the contour plot
    cax : matplotlib.axes.Axes, optional
        Axes object for colorbar
    label : str, optional
        Label for colorbar
    """
    import numpy as np
    from matplotlib.colors import to_rgba, LinearSegmentedColormap

    # Stack probabilities and clean data
    p_stack = np.array([p_s_x[sim].T for sim in sim_name])
    p_stack = np.nan_to_num(p_stack, nan=0.0)
    p_stack /= np.max(p_stack)

    # Find dominant simulation in each cell
    max_idx = np.argmax(p_stack, axis=0)

    # Create meshgrid for contour coordinates
    # Note: assumes 100x100 mesh (p_stack.shape = [n_sims, 100, 100])
    x = np.arange(p_stack.shape[2]) * 10 / 99
    y = np.arange(p_stack.shape[1]) * 10 / 99
    X, Y = np.meshgrid(x, y)

    # Contour levels
    levels = np.linspace(0.001, 1, 10)

    # Plot contours for each simulation
    for i, sim in enumerate(sim_name):
        mask = max_idx == i
        P_masked = np.where(mask, p_stack[i], 0)

        if np.any(mask):
            # Create custom colormap: transparent to base color
            base_rgba = to_rgba(colors[sim])
            cmap = LinearSegmentedColormap.from_list(
                f"{sim}_cmap",
                [(0, (base_rgba[0], base_rgba[1], base_rgba[2], 0)), (1, base_rgba)],
            )

            cs = ax.contourf(
                X,
                Y,
                P_masked,
                levels=levels,
                cmap=cmap,
                extend="max",
                extent=[0, 10, 0, 10],
            )

    # Add colorbar if requested
    if cax is not None and label is not None:
        # Use a simple colormap for the colorbar
        from matplotlib.colors import Normalize
        from matplotlib import cm

        norm = Normalize(vmin=0, vmax=1)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), cax=cax)
        cbar.set_label(label, fontsize=20)
        cbar.ax.tick_params(labelsize=20)


def plot_entropy_contours(
    entropy, fig, ax, cax=None, cmap=cm.viridis, extend="neither", label=None
):
    """
    Plot Shannon entropy contours.

    Parameters
    ----------
    entropy : numpy.ndarray
        2D array of entropy values (100x100 grid)
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object for the contour plot
    cax : matplotlib.axes.Axes, optional
        Axes object for colorbar
    cmap : matplotlib.colors.Colormap, optional
        Colormap for contour plot
    extend : str, optional
        How to extend the color scale (e.g., 'neither', 'both', 'min', 'max')
    label : str, optional
        Label for colorbar
    """

    # Create meshgrid for contour coordinates
    # Note: assumes 100x100 mesh
    x = np.arange(entropy.shape[1]) * 10 / 99
    y = np.arange(entropy.shape[0]) * 10 / 99
    X, Y = np.meshgrid(x, y)

    # Set contour levels - entropy ranges from 0 to log2(7) ≈ 2.807
    max_entropy = np.log2(7)  # Maximum possible entropy for 7 simulations
    levels = np.linspace(0, max_entropy, 20)

    # Plot filled contours
    cs = ax.contourf(
        X,
        Y,
        entropy,
        levels=levels,
        cmap=cmap,
        extend=extend,
        extent=[0, 10, 0, 10],
    )

    # Add contour lines for better readability
    cs_lines = ax.contour(
        X,
        Y,
        entropy,
        levels=levels[::3],  # Fewer contour lines
        colors="white",
        linewidths=0.5,
        alpha=0.7,
        extent=[0, 10, 0, 10],
    )

    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    # Add colorbar if requested
    if cax is not None and label is not None:
        norm = Normalize(vmin=0, vmax=max_entropy)
        cbar = fig.colorbar(cs, cax=cax)
        cbar.set_label(label, fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        # Add entropy interpretation labels
        cbar.ax.text(
            0.1,
            -0.06,
            "Low (certain)",
            transform=cbar.ax.transAxes,
            fontsize=12,
            ha="left",
            va="center",
        )
        cbar.ax.text(
            0.1,
            1.03,
            "High (uncertain)",
            transform=cbar.ax.transAxes,
            fontsize=12,
            ha="left",
            va="center",
        )
