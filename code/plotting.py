"""Shared plotting helpers for brain visualizations."""

GHOST_NEURON_ALPHA = 0.10
GHOST_NEURON_DOT_SIZE = 1.5
ACTIVE_NEURON_DOT_SIZE = 10
HIGHLIGHT_NEURON_DOT_SIZE = 16


def plot_brain_ghost(ax, coords):
    """Draw every neuron as a faint background reference layer."""
    ghost = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c="black",
        alpha=GHOST_NEURON_ALPHA,
        s=GHOST_NEURON_DOT_SIZE,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    ax.set_aspect("equal", adjustable="box")
    return ghost
