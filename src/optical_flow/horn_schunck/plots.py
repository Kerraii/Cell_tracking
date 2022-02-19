import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause, gca
from pathlib import Path

from src.optical_flow.utils import get_arrows


def plotderiv(fx, fy, ft, save_location: Path, fn: Path):
    fg = figure(figsize=(18, 5))
    ax = fg.subplots(1, 3)

    for f, a, t in zip((fx, fy, ft), ax, ("$f_x$", "$f_y$", "$f_t$")):
        h = a.imshow(f, cmap="bwr")
        a.set_title(t)
        fg.colorbar(h, ax=a)
    fg.savefig(Path(save_location, "derivs", fn))
    plt.close(fg)


def compareGraphs(u, v, Iold, blobs, scale: int = 2, title: str = "",
                  fn: Path = None, save_location: Path = None, full_stats: bool = False):
    """
    makes quiver
    """

    ax = figure(dpi=300).gca()
    ax.imshow(Iold, cmap="gray", origin="lower")
    ax.xaxis.tick_top()
    ax.set_ylim([0, 480])
    ax.set_xlim([0, 640])
    ax.invert_yaxis()
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    if full_stats:
        for i in range(0, u.shape[0], 5):
            for j in range(0, v.shape[1], 5):
                if v[i, j] > 0.1 and u[i, j] > 0.1 and i % 2 == 0 and j % 2 == 0:
                    ax.arrow(
                        j,
                        i,
                        v[i, j] * scale,
                        u[i, j] * scale,
                        color="red",
                        head_width=3.5,
                        head_length=1
                    )
    else:  # Enkel gemiddelde pijl tekenen per cell
        arrows = get_arrows(blobs, u, v)
        for arrow in arrows:
            x, y, mean_x, mean_y = arrow
            ax.arrow(
                x,  # X value of center
                y,  # Y value of center
                mean_x * scale,
                mean_y * scale,
                color="red",
                head_width=3 * scale,
                head_length=2 * scale
            )
    ax.set_title(title)

    plt.savefig(Path(save_location, "arrows", fn))
    plt.close()


def plotGuessVsActual(arrows, linked, frame_prev, frame_next, title, save_location, fn):
    ax = figure(dpi=300).gca()
    ax.imshow(frame_prev, cmap="Purples", origin="lower", alpha=0.7)
    ax.imshow(frame_next, cmap="Greens", origin="lower", alpha=0.3)
    ax.xaxis.tick_top()
    ax.set_ylim([0, 480])
    ax.set_xlim([0, 640])
    ax.invert_yaxis()
    for i, (x, y, dx, dy) in enumerate(arrows):
        color = "green" if linked[i][1] is not None else "red"
        ax.arrow(
            x,
            y,
            dx,
            dy,
            color=color,
            head_width=3,
            head_length=2
        )
    ax.set_title(title)

    plt.savefig(Path(save_location, "arrows", fn))
    plt.close()
