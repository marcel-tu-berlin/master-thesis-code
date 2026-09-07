"""One visual language for every figure in eval.plots.

The figures go into a thesis, so they are read next to each other and printed.
That makes three things house style rather than per-figure taste:

- **One palette, colour-blind safe.** Okabe-Ito plus two neutrals. Deuteranopia
  is common enough that a red/green arm pair is a real risk in a printed thesis,
  and an arm keeps its colour across every figure it appears in.
- **One meaning per colour.** Green is a correct episode, vermillion a wrong one,
  grey a reference the arms are compared against. A colour never means "correct"
  in one panel and "arm 4" in the next.
- **Ink only where it carries data.** No top or right spine, no box around a
  legend, a grid light enough to read through. Everything removed here is
  something the reader would otherwise have to look past.

`apply()` is called once, at import of eval.plots. It sets rcParams and nothing
else, so a figure built by hand in a notebook gets the same look for free.
"""

from cycler import cycler

# Okabe-Ito, ordered so the first four (the common case) are maximally separated
# under both deuteranopia and greyscale. The last ones extend it for a seven-arm
# sweep; yellow is left out because it disappears on white.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#4D4D4D",  # dark grey
    "#8C564B",  # brown
    "#7570B3",  # violet
    "#8C8C8C",  # mid grey
]

# Semantic colours. These say what a mark means, so they are never taken from
# the cycle - a green bar is a correct episode in every figure that has one.
CORRECT = "#009E73"
WRONG = "#D55E00"
PRIMARY = "#0072B2"
COST = "#D55E00"
# Reference lines, baselines and annotations: present, never competing.
MUTED = "#6E6E6E"
RULE = "#1A1A1A"
INK = "#1A1A1A"
FAINT = "#DCDCDC"

# Marker shapes for the dose-response series. Two conditions on one axis have to
# stay apart in a greyscale print, where colour alone does not survive.
MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply() -> None:
    """Install the house style into matplotlib's rcParams."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            # DejaVu ships with matplotlib, so the Mac, the GPU box and CI all
            # render the same glyphs. A system font would not.
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.0,
            "figure.facecolor": "white",
            "figure.titlesize": 12.0,
            "savefig.facecolor": "white",
            # 200 is the print floor; below it a thin grid line aliases away.
            "savefig.dpi": 200,
            # Left-aligned titles line up with the y-axis label column, so a grid
            # of panels reads down the left edge instead of zig-zagging.
            "axes.titlesize": 10.0,
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.titlepad": 7.0,
            "axes.labelsize": 9.0,
            "axes.labelcolor": "#3C3C3C",
            "axes.edgecolor": "#AFAFAF",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Bars and markers sit on top of the grid, never under it.
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=PALETTE),
            "grid.color": FAINT,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": "#5A5A5A",
            "ytick.color": "#5A5A5A",
            "xtick.labelcolor": "#3C3C3C",
            "ytick.labelcolor": "#3C3C3C",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.6,
            "legend.columnspacing": 1.3,
            "legend.borderaxespad": 0.2,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "lines.markeredgewidth": 0.0,
            "patch.linewidth": 0.0,
            "errorbar.capsize": 2.5,
        }
    )


def color(i: int) -> str:
    """Palette colour for series `i`, wrapping."""
    return PALETTE[i % len(PALETTE)]


def marker(i: int) -> str:
    """Marker shape for series `i`, wrapping."""
    return MARKERS[i % len(MARKERS)]


def figure_legend(fig, handles, labels, ncol_max: int = 6) -> None:
    """One legend for the whole figure, below the panels.

    A per-panel legend in a grid of nine panels is the same key drawn nine times,
    and it lands on top of the data in whichever panel drew it. Hoisting it out
    gives every panel its full area back.

    Below and not above: constrained layout gives the suptitle and an outside
    upper legend the same strip, and they overprint each other.
    """
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(ncol_max, len(handles)),
    )


def on_color(hex_color: str) -> str:
    """Readable text colour for a label drawn on top of `hex_color`."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.62 else INK
