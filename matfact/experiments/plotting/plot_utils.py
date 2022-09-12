def set_fig_size(width=None, height=None, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Args:
        width (float): Document textwidth or columnwidth in pts
        fraction (float, optional): Fraction of the width which
            you wish the figure to occupy

    Returns:
        fig_dim (tuple): Dimensions of figure in inches
    """

    if width is not None:

        if width == "beamer":
            width_pt = 307.28987
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


def set_arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
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
    ax.arrow(
        xmin,
        ymin,
        xmax - xmin + 0.05 * (xmax - xmin),
        0.0,
        fc="k",
        ec="k",
        lw=1,
        head_width=hw,
        head_length=hl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )

    ax.arrow(
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
