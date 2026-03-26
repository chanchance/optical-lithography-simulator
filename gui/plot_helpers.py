"""Shared matplotlib helpers for GUI panels."""
from gui import theme


def style_ax(ax, title: str = "", image_panel: bool = False) -> None:
    """Apply Toss-style formatting to a matplotlib Axes."""
    ax.set_facecolor(theme.BG_SECONDARY)
    if title:
        ax.set_title(title, fontsize=theme.MPL_TITLE, color=theme.TEXT_PRIMARY,
                     fontweight='600', pad=6)
    if image_panel:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    else:
        ax.tick_params(labelsize=theme.MPL_TICK, colors=theme.TEXT_TERTIARY)
        ax.xaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.yaxis.label.set_color(theme.TEXT_SECONDARY)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(theme.BORDER)
        ax.grid(True, color=theme.MPL_GRID, linewidth=0.5, alpha=0.7, zorder=0)
