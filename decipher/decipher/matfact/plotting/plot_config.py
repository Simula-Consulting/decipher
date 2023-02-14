import matplotlib as mpl


def setup():
    "Plotting standards"

    # Set matplotlib backend.
    mpl.use("Agg")

    nice_fonts = {
        # "text.usetex": True,
        # "font.family": 'Libertine',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        "lines.linewidth": 1.0,
        "axes.titlesize": 12,
        "lines.markersize": 3,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    mpl.rcParams.update(nice_fonts)
