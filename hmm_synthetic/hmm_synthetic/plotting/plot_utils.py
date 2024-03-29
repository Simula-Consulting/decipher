import matplotlib.pyplot as plt


def set_fig_size(
    width: float | str,
    height=None,
    fraction: float = 1.0,
    subplots: tuple[int, int] = (1, 1),
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Args:
        width (float): Document textwidth or columnwidth in pts
        fraction (float, optional): Fraction of the width which
            you wish the figure to occupy

    Returns:
        fig_dim (tuple): Dimensions of figure in inches
    """
    if isinstance(width, str):
        if width == "beamer":
            width_pt = 307.28987
        else:
            raise ValueError(f"{width} is not a valid preset!")
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_arrowed_spines(fig: plt.Figure, ax: plt.Axes) -> None:
    """Create arrows for the x- and y-axis."""

    xmin, xmax = ax.get_xlim()  # type: ignore
    ymin, ymax = ax.get_ylim()  # type: ignore

    # removing the default axis on all sides:
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_visible(False)  # type: ignore

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()  # type: ignore
    bbox = ax.get_window_extent().transformed(dps)  # type: ignore
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1.0 / 40.0 * (ymax - ymin)
    hl = 1.0 / 40.0 * (xmax - xmin)
    lw = 1  # axis line width
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    ax.arrow(  # type: ignore
        xmin,
        ymin,
        xmax - xmin + 0.05 * (xmax - xmin),
        0.0,
        fc="k",
        ec="k",
        lw=lw,
        head_width=hw,
        head_length=hl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )

    ax.arrow(  # type: ignore
        xmin,
        ymin,
        0.0,
        ymax - ymin + 0.05 * (ymax - ymin),
        fc="k",
        ec="k",
        lw=1,
        head_width=yhw,
        head_length=yhl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )
