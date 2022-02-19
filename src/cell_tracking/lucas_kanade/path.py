import numpy as np
import src.cell_tracking.linker as linker


def construct_paths(framedata):
    print("Constructing paths")
    cellpaths = []
    for i, (frameflow, nextblobs) in enumerate(framedata):
        # convert coordinates to arrows
        arrows = np.copy(frameflow)[:, :4]
        arrows[:, 2] = arrows[:, 2]-arrows[:, 0]
        arrows[:, 3] = arrows[:, 3]-arrows[:, 1]
        corrected_flow = linker.link(nextblobs, arrows)
        for (fromx, fromy), to, size in corrected_flow:
            if to is None:
                continue
            tox, toy = to
            for j, (path, firstframe, lastframe) in enumerate(cellpaths):
                x, y, s = path[-1]
                if lastframe == i-1 and fromx == x and fromy == y:
                    path.append([tox, toy, size])
                    cellpaths[j][2] += 1
                    break
            else:
                cellpaths.append([[[fromx, fromy, size], [tox, toy, size]], i, i])
    return cellpaths